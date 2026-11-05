"""Mobile-Wallet Fraud Detection: Streamlit app."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))              # lets joblib find src.features
from src.features import add_features      # noqa: E402

FEATURES = ["type", "amount", "oldbalanceOrg", "oldbalanceDest", "hour"]
NUMERIC = ["amount", "oldbalanceOrg", "oldbalanceDest", "hour"]
TYPES = ["TRANSFER", "CASH_OUT"]
MAX_ROWS = 100_000

st.set_page_config(page_title="Mobile-Wallet Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(ROOT / "models" / "fraud_pipeline.joblib")
    pipe.set_params(model__n_jobs=1)
    config = json.loads((ROOT / "models" / "config.json").read_text())
    curve = pd.read_csv(ROOT / "models" / "cost_curve.csv")
    return pipe, config, curve


def decide(p, flag_at):
    block_at = max(0.5, flag_at)
    if p >= block_at:
        return "BLOCK"
    if p >= flag_at:
        return "REVIEW"
    return "ALLOW"


def prepare_csv(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "hour" not in df.columns and "step" in df.columns:
        df["hour"] = pd.to_numeric(df["step"], errors="coerce") % 24
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing) +
                         ". Needed: type, amount, oldbalanceOrg, oldbalanceDest and hour (or step).")
    notes = []
    n0 = len(df)
    df = df[df["type"].isin(TYPES)].copy()
    if len(df) < n0:
        notes.append(f"{n0 - len(df)} rows skipped: only TRANSFER and CASH_OUT are scored.")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n1 = len(df)
    df = df.dropna(subset=NUMERIC)
    ok = (df[["amount", "oldbalanceOrg", "oldbalanceDest"]] >= 0).all(axis=1) & df["hour"].between(0, 23)
    df = df[ok].copy()
    if len(df) < n1:
        notes.append(f"{n1 - len(df)} rows skipped: missing, negative or out-of-range values.")
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()
        notes.append(f"Only the first {MAX_ROWS:,} rows were scored.")
    if df.empty:
        raise ValueError("No valid rows to score.")
    return df, notes


try:
    pipe, config, curve = load_artifacts()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

st.title("🛡️ Mobile-Wallet Fraud Detection")
st.caption("Trained on PaySim, a simulated mobile-money dataset. A demo of a screening tool, "
           "not a production fraud system.")

st.sidebar.header("Decision settings")
flag_at = st.sidebar.slider("Flag threshold", 0.01, 0.99, float(config["threshold"]), 0.01,
                            help="Transactions with a fraud probability at or above this are flagged.")
c_fp = st.sidebar.number_input("Cost of one false alarm (currency units)", min_value=1.0,
                               value=float(config["c_fp"]), step=50.0)
st.sidebar.caption("Scores at or above 0.50 are blocked. Scores between the flag threshold "
                   "and 0.50 go to manual review.")

tab1, tab2, tab3, tab4 = st.tabs(["Check one transaction", "Score a CSV",
                                  "Threshold and cost", "About"])

with tab1:
    preset = st.selectbox("Start from an example", ["Custom", "Fraud-like pattern", "Normal-looking cash-out"])
    d = {"Custom": ("TRANSFER", 50_000.0, 100_000.0, 20_000.0, 14),
         "Fraud-like pattern": ("TRANSFER", 181_000.0, 181_000.0, 0.0, 4),
         "Normal-looking cash-out": ("CASH_OUT", 4_000.0, 90_000.0, 30_000.0, 15)}[preset]
    left, right = st.columns(2)
    ttype = left.selectbox("Transaction type", TYPES, index=TYPES.index(d[0]), key=f"t_{preset}")
    amount = left.number_input("Amount", min_value=0.0, value=d[1], step=1000.0, key=f"a_{preset}")
    hour = left.slider("Hour of day (0 to 23)", 0, 23, d[4], key=f"h_{preset}")
    old_org = right.number_input("Sender's balance before", min_value=0.0, value=d[2], step=1000.0, key=f"o_{preset}")
    old_dest = right.number_input("Receiver's balance before", min_value=0.0, value=d[3], step=1000.0, key=f"d_{preset}")

    if st.button("Check transaction", type="primary"):
        if amount <= 0:
            st.error("Please enter an amount greater than 0.")
        else:
            try:
                row = pd.DataFrame([{"type": ttype, "amount": amount, "oldbalanceOrg": old_org,
                                     "oldbalanceDest": old_dest, "hour": hour}])
                p = float(pipe.predict_proba(row)[0, 1])
                verdict = decide(p, flag_at)
                st.metric("Fraud probability", f"{p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))
                if verdict == "BLOCK":
                    st.error("BLOCK: very likely fraud.")
                elif verdict == "REVIEW":
                    st.warning("REVIEW: suspicious, send to a manual check.")
                else:
                    st.success("ALLOW: looks normal.")
                f = add_features(row).iloc[0]
                st.markdown("**What the model looked at**")
                st.markdown(f"- Sender's account emptied: **{'yes' if f['drains_account'] else 'no'}**")
                st.markdown(f"- Amount compared with the sender's balance: **{f['amount_to_balance']:.0%}**")
                st.markdown(f"- Receiver's account was empty before: **{'yes' if f['dest_was_empty'] else 'no'}**")
                st.markdown(f"- Hour of day: **{hour}:00**")
            except Exception as exc:
                st.error(f"Could not score this transaction: {exc}")

with tab2:
    st.write("Upload a CSV with the columns **type, amount, oldbalanceOrg, oldbalanceDest** "
             "and **hour** (or **step**). Only TRANSFER and CASH_OUT rows are scored.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        try:
            data, notes = prepare_csv(pd.read_csv(file))
            for n in notes:
                st.info(n)
            data["fraud_probability"] = pipe.predict_proba(data[FEATURES])[:, 1].round(4)
            data["decision"] = [decide(p, flag_at) for p in data["fraud_probability"]]
            counts = data["decision"].value_counts()
            a, b, c = st.columns(3)
            a.metric("Blocked", int(counts.get("BLOCK", 0)))
            b.metric("Sent to review", int(counts.get("REVIEW", 0)))
            c.metric("Allowed", int(counts.get("ALLOW", 0)))
            st.write("Highest-risk transactions")
            st.dataframe(data.sort_values("fraud_probability", ascending=False).head(25))
            st.download_button("Download all scored rows", data.to_csv(index=False).encode(),
                               "scored_transactions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

with tab3:
    st.write("Missed fraud is counted as the stolen amount; each false alarm costs the value "
             "set in the sidebar. These figures come from out-of-fold predictions on the "
             "training data (about 2.2 million transactions).")
    c = curve.copy()
    c["total_cost"] = c["missed_fraud_amount"] + c_fp * c["false_alarms"]
    best = c.loc[c["total_cost"].idxmin()]
    here = c.loc[(c["threshold"] - flag_at).abs().idxmin()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Missed fraud amount", f"{here['missed_fraud_amount']:,.0f}")
    m2.metric("False alarms", f"{int(here['false_alarms']):,}")
    m3.metric("Total cost", f"{here['total_cost']:,.0f}")
    st.write(f"With this false-alarm cost, the cheapest threshold is **{best['threshold']:.2f}**.")
    plot = c[c["threshold"] <= 0.9].set_index("threshold")[["total_cost"]] / 1e6
    st.line_chart(plot)
    st.caption("Total cost in millions, thresholds up to 0.9 shown.")

with tab4:
    t = config["test"]
    st.markdown(f"""
**Model:** tuned random forest inside one preprocessing pipeline.

**Test results** ({t['rows']:,} transactions, {t['frauds']:,} frauds, never used for training or tuning):
PR-AUC {t['pr_auc']}, ROC-AUC {t['roc_auc']}, precision {t['precision_at_0.5']} and recall {t['recall_at_0.5']} at threshold 0.5.

**Limitations**
- The data is simulated and its fraud is unusually regular, so real-world performance would likely be lower.
- The cost of a false alarm is an assumption you can change in the sidebar.
- Fraud tactics change, so a real system needs regular retraining and monitoring.

Dataset: PaySim, https://www.kaggle.com/datasets/ealaxi/paysim1
""")