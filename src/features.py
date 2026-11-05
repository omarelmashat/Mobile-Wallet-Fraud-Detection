def add_features(X):
    X = X.copy()
    X["drains_account"] = ((X["oldbalanceOrg"] > 0) & (X["amount"] >= X["oldbalanceOrg"])).astype(int)
    X['amount_to_balance'] = X['amount'] / (X['oldbalanceOrg'] + 1)
    X['dest_was_empty'] = (X['oldbalanceDest'] == 0).astype(int)
    return X
