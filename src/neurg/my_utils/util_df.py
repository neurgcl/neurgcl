import pandas as pd

def df_desc(array, percentiles=[0.25, 0.5, 0.75, 0.95, 0.99], **kwargs):
    df = pd.DataFrame(array).describe(percentiles).transpose()
    for key, val in kwargs.items():
        df[key] = val
    return df


def reorder_columns(df: pd.DataFrame, first_cols=[]) -> pd.DataFrame:
    cols_new = list(df.columns)
    for item in first_cols:
        if item in cols_new:
            cols_new.remove(item)
    cols_new = first_cols + cols_new
    return df[cols_new].copy()
