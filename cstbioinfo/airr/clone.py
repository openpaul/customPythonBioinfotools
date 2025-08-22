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
        .drop("light_rank", "heavy_rank")
    )
