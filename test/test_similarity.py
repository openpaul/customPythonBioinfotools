import polars as pl

from cstbioinfo.immune import clone_overlap


def test_overlap():
    v_genes_1 = ["IGHV1-1", "IGLV1-2", "IGKV1-3"]
    v_genes_2 = ["IGHV1-1", "IGLV1-2", "IGKV1-4"]
    j_genes_1 = ["IGHJ1", "IGLJ1", "IGKJ1"]
    j_genes_2 = ["IGHJ1", "IGLJ2", "IGKJ1"]
    gene_counts_1 = [10, 5, 2]
    gene_counts_2 = [10, 5, 3]

    df = pl.DataFrame(
        {
            "v_call": v_genes_1 + v_genes_2,
            "j_call": j_genes_1 + j_genes_2,
            "gene_counts": gene_counts_1 + gene_counts_2,
            "sample_id": ["sample1"] * len(v_genes_1) + ["sample2"] * len(v_genes_2),
        }
    )
    result = clone_overlap(
        df,
        feature_columns=["v_call", "j_call"],
        count_column="gene_counts",
        sample_column="sample_id",
        fraction=True,
    )
    assert isinstance(result, pl.DataFrame)

    assert "overlap" in result.columns
    assert result.get_column("overlap").max() == 1.0
    assert result.get_column("overlap").min() == 1 / 3
