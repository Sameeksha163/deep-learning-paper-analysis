import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data(dataset):
    if dataset == "california_housing":
        data = fetch_california_housing(as_frame=True)
        df = pd.concat([data.data, data.target.rename("target")], axis=1)
        print("Loaded California Housing dataset.")
        return df
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
