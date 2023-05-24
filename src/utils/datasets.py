from sklearn.datasets import fetch_openml

ACS_INCOME_ID = 43141
IBA_DEPRESSION_ID = 45040


def fetch_openml_data(data_id):
    data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    X = data.data
    y_true = data.target
    data_dict = {"features": X, "labels": y_true}
    return data_dict
