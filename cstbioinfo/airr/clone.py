import polars as pl


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
