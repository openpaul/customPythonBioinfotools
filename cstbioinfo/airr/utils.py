import pandas as pd
import polars as pl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def _translate(sequence: str | SeqRecord | Seq) -> str:
    if isinstance(sequence, SeqRecord):
        sequence = str(sequence.seq)
    seq = sequence.replace("-", "").replace(".", "").replace(" ", "")
    if len(seq) % 3 != 0:
        raise ValueError("Sequence length is not a multiple of 3.")
    return str(Seq(seq).translate())


def translate(
    df: pd.DataFrame | pl.DataFrame, column: str = "sequence_alignment"
) -> pd.DataFrame | pl.DataFrame:
    """
    Translate nucleotide sequences in a DataFrame column to amino acid sequences.

    This does very simple translation without checking for start/stop codons.

    Args:
        df: Input DataFrame (pandas or polars)
        column: Name of the column containing nucleotide sequences
    Returns:
        DataFrame with an additional column containing translated amino acid sequences

    Example:
        ```python
        import polars as pl
        from cstbioinfo.airr import translate
        df = pl.DataFrame({"sequence_alignment": ["ATGGCC", "ATGCGT"]})
        df_translated = translate(df, column="sequence_alignment")
        print(df_translated)
        ```
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame.")

    sequences = []
    if isinstance(df, pd.DataFrame):
        sequences = df[column]
    elif isinstance(df, pl.DataFrame):
        sequences = df.get_column(column)
    else:
        raise ValueError("Input must be a pandas or polars DataFrame.")

    translated_seqs = [_translate(seq) for seq in sequences]
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df[column + "_aa"] = translated_seqs
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(translated_seqs).alias(column + "_aa"))
    return df


def call2gene(df: pl.DataFrame) -> pl.DataFrame:
    # makes all call to gene colums
    # eg v_call -> v_gene by stripping after first *

    for col in df.columns:
        if col.endswith("_call"):
            gene_col = col.replace("_call", "_gene")
            df = df.with_columns(
                pl.col(col).str.split("*").list.first().alias(gene_col)
            )
    return df
