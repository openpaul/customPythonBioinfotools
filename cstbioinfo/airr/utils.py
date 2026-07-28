import pandas as pd
import polars as pl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# AIRR schema required fields per the AIRR Community standard
# https://docs.airr-community.org/en/stable/datarep/rearrangements.html
_AIRR_REQUIRED_COLUMNS = {
    "sequence_id",
    "sequence",
    "rev_comp",
    "productive",
    "v_call",
    "d_call",
    "j_call",
    "sequence_alignment",
    "germline_alignment",
    "junction",
    "junction_aa",
    "np1_length",
    "np2_length",
    "v_cigar",
    "d_cigar",
    "j_cigar",
}

_VALID_LOCUS_VALUES = {"IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"}


def validate_airr(
    df: pd.DataFrame | pl.DataFrame,
    required_columns: set[str] | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """
    Validate an AIRR-seq DataFrame against the AIRR Community standard.

    Checks for required columns, unique sequence IDs, valid locus values,
    and boolean-compatible productive/rev_comp fields.

    Args:
        df: Input DataFrame (pandas or polars).
        required_columns: Override the set of required column names. Defaults
            to the standard AIRR required columns.
        raise_on_error: If True, raise a ValueError listing all violations.
            If False, return the list of error messages instead.

    Returns:
        A list of validation error messages (empty if the DataFrame is valid).

    Raises:
        ValueError: If raise_on_error is True and any validation errors exist.

    Example:
        ```python
        import polars as pl
        from cstbioinfo.airr import validate_airr
        df = pl.read_csv("airr_table.tsv", separator="\\t")
        validate_airr(df)
        ```
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("df must be a pandas or polars DataFrame.")

    errors: list[str] = []
    columns = set(df.columns)
    required = required_columns if required_columns is not None else _AIRR_REQUIRED_COLUMNS

    # 1. Required columns
    missing = required - columns
    if missing:
        errors.append(f"Missing required AIRR columns: {sorted(missing)}")

    # 2. sequence_id uniqueness
    if "sequence_id" in columns:
        if isinstance(df, pl.DataFrame):
            n_total = df.height
            n_unique = df["sequence_id"].n_unique()
        else:
            n_total = len(df)
            n_unique = df["sequence_id"].nunique()
        if n_unique != n_total:
            errors.append(
                f"'sequence_id' is not unique: {n_total - n_unique} duplicate(s) found."
            )

    # 3. locus values (optional column, but validated when present)
    if "locus" in columns:
        if isinstance(df, pl.DataFrame):
            invalid_loci = (
                df["locus"]
                .drop_nulls()
                .filter(~pl.Series(df["locus"].drop_nulls()).is_in(list(_VALID_LOCUS_VALUES)))
                .unique()
                .to_list()
            )
        else:
            invalid_loci = sorted(
                df["locus"].dropna()[~df["locus"].dropna().isin(_VALID_LOCUS_VALUES)].unique()
            )
        if invalid_loci:
            errors.append(
                f"Invalid 'locus' values found: {invalid_loci}. "
                f"Expected one of {sorted(_VALID_LOCUS_VALUES)}."
            )

    # 4. productive / rev_comp should be boolean or string-boolean
    for bool_col in ("productive", "rev_comp"):
        if bool_col not in columns:
            continue
        if isinstance(df, pl.DataFrame):
            dtype = df[bool_col].dtype
            is_valid = dtype == pl.Boolean or dtype in (pl.Utf8, pl.String)
        else:
            dtype = df[bool_col].dtype
            is_valid = str(dtype) in ("bool", "object")
        if not is_valid:
            errors.append(
                f"Column '{bool_col}' should be boolean or string type, got {dtype}."
            )

    if raise_on_error and errors:
        raise ValueError("AIRR DataFrame validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return errors


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
