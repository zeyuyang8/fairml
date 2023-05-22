import pandas as pd
from sklearn.datasets import fetch_openml


def fetch_openml_dataset(choice):
    """Load dataset according to the choice from OpenML.

    Args:
        choice (str): dataset name

    Returns:
        X (pd.Dataframe): features
        y_true (pd.Series): true labels
    """
    if choice == "ACSincome":
        data = fetch_openml(data_id=43141, as_frame=True, parser='auto')
    elif choice == "IBADepression":
        data = fetch_openml(data_id=45040, as_frame=True, parser='auto')

    X = pd.get_dummies(data.data)
    y_true = data.target
    return {"features": X, "labels": y_true}
