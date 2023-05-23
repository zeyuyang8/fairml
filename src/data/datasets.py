import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fetch_openml_dataset(choice, sensitive_feature=None):
    """Load dataset according to the choice from OpenML.

    Args:
        choice (str): dataset name
        sensitive_feature (str or list of str): sensitive feature

    Returns:
        X (pd.Dataframe): features
        y_true (pd.Series): true labels
    """
    if choice == "ACSincome":
        data = fetch_openml(data_id=43141, as_frame=True, parser='auto')
        X = data.data
        y_true = data.target
    elif choice == "IBADepression":
        data = fetch_openml(data_id=45040, as_frame=True, parser='auto')
        X = data.data
        y_true = data.target
    elif choice == "UCIadult":
        data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
        X = pd.get_dummies(data.data)
        columns = X.columns
        y_true = (data.target == '>50K') * 1
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = pd.DataFrame(X, columns=columns)
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)

    if sensitive_feature:
        sensitive = data.data[sensitive_feature]

    return {"features": X, "labels": y_true, "sensitive": sensitive}
