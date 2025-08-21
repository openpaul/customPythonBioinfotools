from typing import Union
import numpy as np
import polars as pl
import pandas as pd

ArrayLike = Union[
    np.ndarray, list[float], tuple[float, ...] | list[int], tuple[int, ...]
]


def _ruzicka(x: ArrayLike, y: ArrayLike, top_n: int | None = None) -> float:
    """
    Calculate the Ruzicka similarity between two arrays.
    Parameters
    ----------
    x : ArrayLike
        First array.
    y : ArrayLike
        Second array.
    top_n : int, optional
        Number of top elements to consider for similarity calculation.
        If None, all elements are considered.
    Returns
    -------
    float
        Ruzicka similarity score between 0 and 1.
    Raises
    ValueError
        If the input arrays are not of the same length or if top_n is invalid.
    """

    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")

    if top_n is not None and (top_n <= 0 or top_n > len(x)):
        raise ValueError(
            "top_n must be a positive integer less than or equal to the length of the input arrays."
        )

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)

    if top_n is not None:
        idx_a = np.argsort(a)[-top_n:]
        idx_b = np.argsort(b)[-top_n:]
        idx = np.union1d(idx_a, idx_b)
        a = a[idx]
        b = b[idx]

    num = np.sum(np.minimum(a, b))
    den = np.sum(np.maximum(a, b))
    print(f"Numerator: {num}, Denominator: {den}")  # Debugging output
    return float(num / den) if den != 0 else 0.0


def ruzicka_similarity(
    df: pl.DataFrame | pd.DataFrame,
    top_n: int | None = None,
    feature_columns: list[str] = ["v_call", "j_call", "junction_aa"],
    count_column: str = "count",
    sample_column: str = "sample_id",
) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    # pairwise Ruzicka similarity for each sample
    if not all(
        col in df.columns for col in feature_columns + [count_column, sample_column]
    ):
        # tell the user which columns are missing
        missing_cols = [
            col
            for col in feature_columns + [count_column, sample_column]
            if col not in df.columns
        ]
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_cols)}")

    results = []
    samples = df.get_column(sample_column).unique().to_list()
    # sort samples
    samples.sort()
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sample_i = samples[i]
            sample_j = samples[j]

            df_subset = (
                df.filter(
                    (pl.col(sample_column) == sample_i)
                    | (pl.col(sample_column) == sample_j)
                )
                .select(feature_columns + [count_column, sample_column])
                .pivot(index=feature_columns, on=sample_column, values=count_column)
            ).fill_null(0)
            if df_subset.height == 0:
                # if no data for this sample pair, skip
                continue

            similarity = _ruzicka(
                df_subset.get_column(sample_i).to_numpy(),
                df_subset.get_column(sample_j).to_numpy(),
                top_n=top_n,
            )
            results.append(
                {
                    "sample_1": sample_i,
                    "sample_2": sample_j,
                    "similarity": similarity,
                }
            )

    if results:
        return pl.DataFrame(results)
    else:
        return pl.DataFrame(
            {
                "sample_1": [],
                "sample_2": [],
                "similarity": [],
            }
        )
