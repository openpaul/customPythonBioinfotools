from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

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

    if top_n is not None and (top_n <= 0):
        raise ValueError("top_n must be a positive integer.")

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)

    if top_n is not None:
        if top_n > len(a):
            top_n = len(a)
        if top_n > len(b):
            top_n = len(b)
        idx_a = np.argsort(a)[-top_n:]
        idx_b = np.argsort(b)[-top_n:]
        idx = np.union1d(idx_a, idx_b)
        a = a[idx]
        b = b[idx]

    num = np.sum(np.minimum(a, b))
    den = np.sum(np.maximum(a, b))
    return float(num / den) if den != 0 else 0.0


def ruzicka_similarity(
    df: pl.DataFrame | pd.DataFrame,
    top_n: int | None = None,
    feature_columns: list[str] = ["v_call", "j_call", "junction_aa"],
    count_column: str = "count",
    sample_column: str = "sample_id",
) -> pl.DataFrame:
    """
    Calculate pairwise Ruzicka similarity between samples in a DataFrame.

    Args:
        df: Input DataFrame containing features, counts, and sample identifiers.
        top_n: Number of top elements to consider for similarity calculation.
               If None, all elements are considered.
        feature_columns: List of columns representing features. This could be the clone id for example.
        count_column: Column name representing the counts or frequencies of the features/clones.
        sample_column: Column name representing the sample identifiers.
    Returns:
        Polars DataFrame with pairwise Ruzicka similarity scores between samples.

    Example:
        >>> import polars as pl
        >>> from cstbioinfo.immune.ruzicka import ruzicka_similarity
        >>> data = {
        ...     "sample_id": ["A", "A", "B", "B", "C"],
        ...     "v_call": ["V1", "V2", "V1", "V3", "V2"],
        ...     "j_call": ["J1", "J1", "J1", "J2", "J1"],
        ...     "junction_aa": ["CAR", "CAG", "CAR", "CAT", "CAG"],
        ...     "count": [10, 5, 8, 7, 3],
        ... }
        >>> df = pl.DataFrame(data)
        >>> similarity_df = ruzicka_similarity(df, top_n=2)
        >>> print(similarity_df.shape)
        (9, 3)

        See the Ruzicka notebook in the Github repository for more examples.
    """
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

    samples = df.get_column(sample_column).unique().to_list()
    # sort samples
    samples.sort()

    results = [
        {"sample_1": sample_name, "sample_2": sample_name, "similarity": 1.0}
        for sample_name in samples
    ]
    comparisons = len(samples) * (len(samples) - 1) // 2
    pbar = tqdm(
        total=comparisons,
        desc="Calculating Ruzicka similarity",
    )
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
                pbar.update(1)
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
            results.append(
                {
                    "sample_1": sample_j,
                    "sample_2": sample_i,
                    "similarity": similarity,
                }
            )
            pbar.update(1)

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
