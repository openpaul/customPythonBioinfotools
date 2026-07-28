import polars as pl

AIRR_REQUIRED_COLUMNS: list[str] = [
    "sequence_id",
    "sequence",
    "locus",
    "v_call",
    "j_call",
    "junction",
    "junction_aa",
]

AIRR_VALID_LOCI: set[str] = {"IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD"}


def validate_airr_df(
    df: pl.DataFrame,
    required_columns: list[str] | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """
    Validate an AIRR-seq DataFrame against required columns and value constraints.

    Args:
        df: Polars DataFrame to validate.
        required_columns: Columns that must be present and non-null. Defaults to
            ``AIRR_REQUIRED_COLUMNS``.
        raise_on_error: If ``True`` (default), raise ``ValueError`` on the first
            error found. If ``False``, return all error messages instead.

    Returns:
        A list of error message strings. Empty list means the DataFrame is valid.

    Raises:
        TypeError: If ``df`` is not a polars DataFrame.
        ValueError: If ``raise_on_error`` is ``True`` and any validation fails.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected a polars DataFrame, got {type(df).__name__}.")

    if required_columns is None:
        required_columns = AIRR_REQUIRED_COLUMNS

    errors: list[str] = []

    # Check required columns are present
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for nulls in required columns that are present
    for col in required_columns:
        if col not in df.columns:
            continue
        null_count = df[col].null_count()
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null value(s).")

    # Check locus column contains only valid values
    if "locus" in df.columns:
        invalid_loci = (
            df["locus"]
            .drop_nulls()
            .filter(~pl.Series(df["locus"].drop_nulls()).is_in(list(AIRR_VALID_LOCI)))
        )
        if len(invalid_loci) > 0:
            errors.append(
                f"Column 'locus' contains invalid values: {invalid_loci.unique().to_list()}. "
                f"Valid values are: {sorted(AIRR_VALID_LOCI)}."
            )

    if raise_on_error and errors:
        raise ValueError("\n".join(errors))

    return errors


def perfect_paired(
    df: pl.DataFrame, cell_column: str = "cell_id", umi_count_column: str = "umi_count"
) -> pl.DataFrame:
    return (
        df.sort([cell_column, umi_count_column], descending=True)
        .with_columns(
            pl.col("locus").is_in(["TRG", "TRA", "IGL", "IGK"]).alias("is_light"),
            pl.col("locus").is_in(["TRB", "IGH", "TRD"]).alias("is_heavy"),
        )
        .filter(pl.col("is_light") | pl.col("is_heavy"))
        .with_columns(
            pl.cum_count(cell_column)
            .over([cell_column, "is_light"])
            .alias("light_rank"),
            pl.cum_count(cell_column)
            .over([cell_column, "is_heavy"])
            .alias("heavy_rank"),
        )
        .filter(
            (
                (pl.col("is_light") & (pl.col("light_rank") == 1))
                | (pl.col("is_heavy") & (pl.col("heavy_rank") == 1))
            )
        )
        .with_columns(
            pl.col(cell_column).len().over(cell_column).alias("chain_count"),
        )
        .filter(pl.col("chain_count") == 2)
        .drop("light_rank", "heavy_rank", "is_light", "is_heavy", "chain_count")
    )


def cast_to_pairs(
    df: pl.DataFrame,
    cell_column: str = "cell_id",
    umi_count_column: str = "umi_count",
    only_pairs: bool = True,
) -> pl.DataFrame:
    if only_pairs:
        df = perfect_paired(
            df, cell_column=cell_column, umi_count_column=umi_count_column
        )

    heavy_prefix = "heavy"
    light_prefix = "light"

    # now we can pivot but for the only_pairs False we need to do all combinations between heavy and light
    if only_pairs:
        return df.with_columns(
            pl.when(pl.col("locus").is_in(["TRB", "IGH", "TRD"]))
            .then(pl.lit(heavy_prefix))
            .otherwise(pl.lit(light_prefix))
            .alias("chain_type")
        ).pivot(index=cell_column, on="chain_type")
    else:
        # this is more complicated as each heavy can pair with each light in each cell.
        # we can use join to make all pairs
        heavy_df = df.filter(pl.col("locus").is_in(["TRB", "IGH", "TRD"])).with_columns(
            pl.lit(heavy_prefix).alias("chain_type")
        )
        light_df = df.filter(
            pl.col("locus").is_in(["TRG", "TRA", "IGL", "IGK"])
        ).with_columns(pl.lit(light_prefix).alias("chain_type"))
        # rename all but cell_column in heavy with _heavy suffix
        heavy_df = heavy_df.rename(
            {col: f"{col}_heavy" for col in heavy_df.columns if col != cell_column}
        )
        light_df = light_df.rename(
            {col: f"{col}_light" for col in light_df.columns if col != cell_column}
        )
        paired_df = heavy_df.join(
            light_df,
            left_on=cell_column,
            right_on=cell_column,
            suffix="_light",
            how="inner",
        ).drop(["chain_type_heavy", "chain_type_light"])
        return paired_df
