# Mobile-Wallet Fraud Detection

A machine-learning demo that scores mobile-money transfers and cash-outs for fraud risk. It uses the simulated PaySim dataset and includes a Streamlit app for checking one transaction, scoring a CSV, and exploring the cost trade-off behind the review threshold.

**Public demo:** Not deployed yet. The app has been run and checked locally.

## Results

The tuned random-forest pipeline was evaluated once on a held-out test set of 554,082 transactions, including 1,643 fraud cases.

| Metric | Result | Evaluation |
|---|---:|---|
| PR-AUC | 0.998 | Test set; majority-class baseline: 0.003 |
| ROC-AUC | 0.9991 | Test set |
| Precision | 0.9994 | Test set, threshold 0.50 |
| Recall | 0.9963 | Test set, threshold 0.50 |
| F1 | 0.9979 | Test set, threshold 0.50 |

These results are unusually strong because PaySim is simulated and its fraud patterns are regular. They should not be treated as expected real-world performance.

## How the app works

- Enter a transaction or upload a CSV to get a fraud probability and decision.
- CSVs need `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, and either `hour` or `step`. Only `TRANSFER` and `CASH_OUT` rows are scored.
- The default review threshold is 0.02. Scores from that threshold up to 0.50 go to review; scores at or above 0.50 are blocked; lower scores are allowed.
- The cost view uses out-of-fold training predictions. It assumes a false alarm costs 100 currency units; this is a configurable assumption, not a value supplied by PaySim.

## Run locally

Python 3.12 is recommended. The saved model was created with scikit-learn 1.6.1, which is pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The sample upload file is `reports/sample_transactions.csv`.

## Data and notebook

The model was trained on [PaySim Mobile Money Simulation](https://www.kaggle.com/datasets/ealaxi/paysim1). The data is simulated; no raw dataset is included in this repository. The Streamlit app uses the saved model and does not need the raw dataset to run.

To rerun the notebook, download the dataset and place it at `data/raw/paysim.csv`. The first notebook cell currently points to a machine-specific absolute path, so update that line to the CSV's location on your computer before running it.

## Project files

```text
app/app.py                         Streamlit app
models/fraud_pipeline.joblib       Trained preprocessing and model pipeline
models/config.json                 Threshold and held-out test metrics
models/cost_curve.csv              Out-of-fold training cost curve
notebooks/fraud_detection.ipynb    Analysis and model development
reports/sample_transactions.csv    Example CSV for the app
requirements.txt                   Pinned app dependencies
src/features.py                    Shared feature engineering
```

## Limitations

PaySim is a simulator, not live wallet data. Its fraud patterns may be easier to detect than fraud in a real service. The false-alarm cost is an assumption, and the chosen threshold should be reviewed with business stakeholders before any real use. This project is a demonstration, not a production fraud-control system.
