from cstbioinfo.immune.ruzicka import _ruzicka
from cstbioinfo.immune import ruzicka_similarity
import pytest
import polars as pl


def test_ruzicka_basic_function():
    # Test with two identical arrays
    x = [1, 2, 3]
    y = [1, 2, 3]
    assert _ruzicka(x, y) == 1.0

    # Test with two completely different arrays
    x = [1, 2, 3, 0, 0, 0]
    y = [0, 0, 0, 1, 2, 3]
    assert _ruzicka(x, y) == 0.0

    # Test with top_n parameter
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    assert pytest.approx(_ruzicka(x, y, top_n=1), 1e-4) == ((1) / (5))

    # Test with unequal lengths
    with pytest.raises(ValueError):
        _ruzicka([1], [1, 2])


def test_ruzicka_similarity():
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
    result = ruzicka_similarity(
        df,
        top_n=2,
        feature_columns=["v_call", "j_call"],
        count_column="gene_counts",
        sample_column="sample_id",
    )
    assert isinstance(result, pl.DataFrame)
    assert "similarity" in result.columns
