from collections.abc import Sequence

import pandas as pd
from sklearn.experimental import (  # noqa: F401 explicitly enable experimental feature in case user requests it
    enable_iterative_imputer,
)
from sklearn.impute._base import _BaseImputer


def impute(data: pd.DataFrame, imputer: _BaseImputer, impute_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Impute missing values in a DataFrame using the provided imputer.

    Args:
        data: DataFrame to impute.
        imputer: Imputer to complete missing values.
        impute_cols: Columns for which to complete missing values. If None, default to all columns.
    """
    if impute_cols is None:
        impute_cols = data.columns

    # Complete missing values
    completed_data = imputer.fit_transform(data)

    # Fill missing values in requested columns with imputed data
    dtypes = data.dtypes[impute_cols]  # Get dtypes of the columns to be imputed
    impute_cols_idx = data.columns.get_indexer(impute_cols)  # Get numerical indices for `impute_cols`
    data[impute_cols] = completed_data[:, impute_cols_idx]
    # Convert imputed columns to their original dtypes, to make sure they are not converted to float
    return data.astype(dtypes)
