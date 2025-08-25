import pandas as pd
import polars as pl
from tqdm import tqdm


def clone_overlap(
    df: pl.DataFrame | pd.DataFrame,
    feature_columns: list[str] = ["v_call", "j_call", "junction_aa"],
    count_column: str = "count",
    sample_column: str = "sample_id",
    fraction: bool = True,
) -> pl.DataFrame:
    """
    Compute the relative overlap coefficient between samples in a DataFrame.

    Clones are defined by passing the feature_columns parameter, which are concatenated
    to form a unique clone identifier. The overlap coefficient is calculated as the size
    of the intersection of clones between two samples divided by the size of the smaller
    sample's clone set.

    Args:
        df (pl.DataFrame | pd.DataFrame): Input DataFrame containing immune repertoire data.
        feature_columns (list of str): List of columns to define a clone (e.g., V gene, J gene, CDR3).
        count_column (str): Column name representing the counts or frequencies of the features/clones.
        sample_column (str): Column name representing the sample identifiers.
        fraction (bool): If True, returns the overlap as a fraction of the smaller sample's
                         clone set size. If False, returns the absolute count of overlapping clones.
    Returns:
        pl.DataFrame: DataFrame with pairwise overlap coefficients between samples.
    Example:
        ```python
        import polars as pl
        from cstbioinfo.immune import clone_overlap
        data = {
            "v_call": ["IGHV1-1", "IGLV1-2", "IGKV1-3", "IGHV1-1", "IGLV1-2", "IGKV1-4"],
            "j_call": ["IGHJ1", "IGLJ1", "IGKJ1", "IGHJ1", "IGLJ2", "IGKJ1"],
            "count": [10, 5, 2, 10, 5, 3],
            "sample_id": ["sample1"] * 3 + ["sample2"] * 3,
        }
        df = pl.DataFrame(data)
        result = clone_overlap(
            df,
            feature_columns=["v_call", "j_call"],
            count_column="count",
            sample_column="sample_id",
            fraction=True,
        )
        ```
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

    if ".clone_definition" in df.columns:
        raise ValueError(
            "DataFrame already contains a column named '.clone_definition'. Please rename or remove this column."
        )

    samples = df.get_column(sample_column).unique().to_list()
    # sort samples
    samples.sort()

    returns = []
    # as scaling is done per sample we need to do all vs all
    comparisons = len(samples) * (len(samples) - 1)
    pbar = tqdm(
        total=comparisons,
        desc="Calculating overlap coefficient",
    )

    df = df.with_columns(
        pl.concat_str(feature_columns, separator="_").alias(".clone_definition")
    )

    for i in range(len(samples)):
        for j in range(len(samples)):
            sample_i = samples[i]
            sample_j = samples[j]
            clones_i = (
                df.filter(pl.col(sample_column) == sample_i)
                .get_column(".clone_definition")
                .to_list()
            )
            clones_j = (
                df.filter(pl.col(sample_column) == sample_j)
                .get_column(".clone_definition")
                .to_list()
            )
            set_i = set(clones_i)
            set_j = set(clones_j)
            intersection = set_i.intersection(set_j)
            if fraction:
                overlap = len(intersection) / len(set_i) if len(set_i) > 0 else 0.0
            else:
                overlap = len(intersection)
            returns.append(
                {
                    "sample_1": sample_i,
                    "sample_2": sample_j,
                    "overlap": overlap,
                }
            )
            pbar.update(1)
    if returns:
        return pl.DataFrame(returns)
    else:
        return pl.DataFrame(
            {
                "sample_1": [],
                "sample_2": [],
                "overlap": [],
            }
        )
