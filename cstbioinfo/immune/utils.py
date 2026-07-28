import gzip
import json
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import requests
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from immunum import Annotator

_NUC_ALPHABET = set("ACGTUN")
_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")


def kabat_sort(positions: list[str]) -> list[str]:
    """
    Sorts residue position strings according to the Kabat numbering scheme.
    Handles optional chain prefixes (e.g., 'H100A', 'L27B') or bare numbers ('100A').

    Insertion order: 100 -> 100A -> 100B -> ... -> 101
    """

    def kabat_key(pos: str):
        cleaned = str(pos).strip()

        # Regex to parse optional chain prefix (H/L), integer position, and insertion code
        match = re.match(
            r"^(?:([A-Za-z])[\-\_]?)?(\d+)(?:[\.\-]?([A-Za-z]))?$", cleaned
        )
        if not match:
            return (2, float("inf"), 0, cleaned)  # Fallback for invalid strings

        chain = match.group(1).upper() if match.group(1) else ""
        base = int(match.group(2))
        insertion = match.group(3)

        # ASCII rank for insertion codes ('A' -> 1, 'B' -> 2, etc.; base without insertion -> 0)
        ins_rank = ord(insertion.upper()) - ord("A") + 1 if insertion else 0

        # Primary key: Chain (L before H if provided, otherwise standard), Base pos, Insertion rank
        chain_order = {"L": 0, "H": 1}.get(chain, 0)

        return (chain_order, base, ins_rank)

    return sorted(positions, key=kabat_key)


def imgt_sort(positions: list[str]) -> list[str]:
    """
    Sorts IMGT position strings according to IMGT unique numbering rules.
    Supports standard positions, decimal insertions (111.1, 112.1), and
    letter insertions (111A, 112A).
    """
    positions = list(set(positions))  # Remove duplicates

    def imgt_key(pos: str):
        # Parses formats like: '104', '111.1', '111A', '112-A', etc.
        match = re.match(r"^(\d+)(?:[\.\-]?([A-Za-z0-9]+))?$", str(pos).strip())
        if not match:
            return (float("inf"), 0, pos)  # Fallback for unexpected formats

        base = int(match.group(1))
        suffix = match.group(2)

        if suffix is None:
            sub = 0
        elif suffix.isdigit():
            sub = float(suffix)
        else:
            # Convert letters/alphanumerics into a numerical rank (A->1, B->2, etc.)
            sub = float(ord(suffix[0].upper()) - ord("A") + 1)

        # IMGT CDR3 insertion rule between 111 and 112:
        # 111 < 111A < 111B ... < 112B < 112A < 112
        if base == 112 and sub > 0:
            return (111.5, -sub)

        return (base, sub)

    return sorted(positions, key=imgt_key)


def _download_file(url: str | Path, dest: str | Path) -> None:
    """
    Download a file from a URL to a specified destination.
    """
    if isinstance(url, Path):
        url = str(url)
    if isinstance(dest, Path):
        dest = str(dest)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)


def _is_url(path: str | Path) -> bool:
    """
    Check if the given path is a URL.
    """
    if isinstance(path, Path):
        path = str(path)
    return (
        path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("ftp://")
    )


def strip_allele(gene_call: str) -> str:
    if "," in gene_call:
        gene_calls = gene_call.split(",")
        # pick the first gene call
        gene_call = gene_calls[0]
    if "*" in gene_call:
        gene_call = gene_call.split("*")[0]
    return gene_call.strip()


def _raw_seq(seq: str) -> str:
    return seq.upper()


def infer_sequence_type(seq: str | SeqRecord) -> str:
    """Infer sequence type as ``nucleotide`` or ``amino_acid``."""
    if isinstance(seq, SeqRecord):
        seq = str(seq.seq)
    clean = _raw_seq(seq)
    if not clean:
        raise ValueError("Sequence is empty.")

    if not re.fullmatch(r"[A-Z]+", clean):
        raise ValueError(
            "Sequence contains unsupported non-letter characters. "
            "Please provide raw nucleotide or amino-acid letters only."
        )

    chars = set(clean)
    if chars.issubset(_NUC_ALPHABET):
        return "nucleotide"
    if chars.issubset(_AA_ALPHABET):
        return "amino_acid"

    raise ValueError(
        "Sequence contains unsupported characters; cannot infer nucleotide or amino acid type."
    )


def find_receptor_boundaries(
    dna_seq: str,
    min_confidence: float = 0.5,
    min_orf_len: int = 20,
    chains: Sequence[str] = ("H", "K", "L", "A", "B", "G", "D"),
    scheme: str = "imgt",
    annotator: Annotator | None = None,
) -> list[dict[str, Any]]:
    """Find receptor-like domains from a nucleotide sequence via 6-frame scanning."""
    raw = _raw_seq(dna_seq)
    if not raw:
        return []
    if not set(raw).issubset(_NUC_ALPHABET):
        invalid = sorted(set(raw) - _NUC_ALPHABET)
        raise ValueError(
            "DNA input contains non-IUPAC nucleotide characters for this function: "
            f"{invalid}."
        )

    seq_obj = Seq(raw)
    seq_len = len(seq_obj)
    if seq_len == 0:
        return []

    anno = annotator or Annotator(
        chains=list(chains), scheme=scheme, min_confidence=min_confidence
    )
    results: list[dict[str, Any]] = []

    frames: dict[str, tuple[str, int, int]] = {}
    for frame in range(3):
        frames[f"forward_{frame + 1}"] = (
            str(seq_obj[frame:].translate(to_stop=False)),
            frame,
            1,
        )
    rev_seq = seq_obj.reverse_complement()
    for frame in range(3):
        frames[f"reverse_{frame + 1}"] = (
            str(rev_seq[frame:].translate(to_stop=False)),
            frame,
            -1,
        )

    for frame_name, (aa_seq, offset, strand) in frames.items():
        if len(aa_seq) <= min_orf_len:
            continue

        result = anno.number(aa_seq)
        chain = getattr(result, "chain", None)
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        q_start = getattr(result, "query_start", None)
        q_end = getattr(result, "query_end", None)

        if chain is None or confidence < min_confidence:
            continue

        if q_start is None or q_end is None:
            numbering = dict(getattr(result, "numbering", {}) or {})
            numbered_len = len(numbering)
            if numbered_len == 0:
                continue
            q_start = 0
            q_end = min(len(aa_seq), numbered_len) - 1

        aa_start = int(q_start)
        aa_end = int(q_end)

        if strand == 1:
            nt_start = (aa_start * 3) + offset
            nt_end = ((aa_end + 1) * 3) + offset - 1
        else:
            nt_start_rev = (aa_start * 3) + offset
            nt_end_rev = ((aa_end + 1) * 3) + offset - 1
            nt_start = seq_len - 1 - nt_end_rev
            nt_end = seq_len - 1 - nt_start_rev

        results.append(
            {
                "frame": frame_name,
                "chain": chain,
                "confidence": round(confidence, 3),
                "numbering": {
                    str(k): str(v)
                    for k, v in dict(getattr(result, "numbering", {}) or {}).items()
                },
                "aa_start": aa_start,
                "aa_end": aa_end,
                "nt_start": int(nt_start),
                "nt_end": int(nt_end),
                "orf_seq": aa_seq,
                "domain_seq": aa_seq[int(q_start) : int(q_end) + 1],
            }
        )

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def extract_receptor_sequence(
    sequence: str | SeqRecord,
    min_confidence: float = 0.5,
    min_orf_len: int = 20,
    chains: Sequence[str] = ("H", "K", "L", "A", "B", "G", "D"),
    scheme: str = "imgt",
) -> dict[str, Any]:
    """Extract a receptor AA domain from amino-acid or nucleotide input.

    For nucleotide input, this uses six-frame receptor boundary detection and
    returns the best-scoring domain. For amino-acid input, it validates by
    numbering directly with immunum.
    """
    seq = str(sequence.seq) if isinstance(sequence, SeqRecord) else sequence
    seq_type = infer_sequence_type(seq)
    anno = Annotator(chains=list(chains), scheme=scheme, min_confidence=min_confidence)

    if seq_type == "nucleotide":
        hits = find_receptor_boundaries(
            seq,
            min_confidence=min_confidence,
            min_orf_len=min_orf_len,
            chains=chains,
            scheme=scheme,
            annotator=anno,
        )
        if not hits:
            raise ValueError("No receptor-like domain found in nucleotide sequence.")
        hit = hits[0]
        return {
            "sequence": str(hit["domain_seq"]),
            "input_type": seq_type,
            "chain": hit.get("chain"),
            "confidence": float(hit.get("confidence", 0.0) or 0.0),
            "numbering": {
                str(k): str(v) for k, v in dict(hit.get("numbering", {}) or {}).items()
            },
            "best_hit": hit,
        }

    aa_seq = seq.upper()
    res = anno.number(aa_seq)
    if getattr(res, "chain", None) is None:
        raise ValueError("Failed to number amino-acid sequence.")
    return {
        "sequence": aa_seq,
        "input_type": seq_type,
        "chain": getattr(res, "chain", None),
        "confidence": float(getattr(res, "confidence", 0.0) or 0.0),
        "numbering": {
            str(k): str(v) for k, v in dict(getattr(res, "numbering", {}) or {}).items()
        },
        "best_hit": None,
    }


def get_oas(path: str | Path) -> pl.DataFrame:
    """
    Load an OAS (Observed Antibody Space) CSV file from a local path or URL
    into a Polars DataFrame.

    Loading OAS files is tricky because the first line is a JSON-like
    dictionary containing metadata, and the rest is a standard CSV. This function
    handles both local files and URLs, and extracts the metadata into
    separate columns in the DataFrame.

    Args:
        path: Local file path or URL to the OAS CSV file (must be gzipped)
    Returns:
        Polars DataFrame containing the OAS data with metadata columns added


    Examples:
        ```python
        from cstbioinfo.immune import get_oas
        df = get_oas("https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Bashford_2013/csv/ERR220451_Heavy_Bulk.csv.gz")
        print(df.head(5))
        ```

        Or with a local file:
        ```python
        df = get_oas("/path/to/local/OAS_file.csv.gz")
        print(df.head(5))
        ```

        If you use this function to access the [OAS database](https://opig.stats.ox.ac.uk/webapps/oas/), please respect their license
        and cite the original publication:

        > Olsen, Tobias H., Fergus Boyles, and Charlotte M. Deane. "Observed Antibody Space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences." Protein Science 31.1 (2022): 141-146.


        I also recommend writing the downloaded file to disk if you plan to use it multiple times,
        to avoid repeated downloads.

        ```python
        from pathlib import Path
        import polars as pl
        from cstbioinfo.immune import get_oas

        local_file = Path("/path/to/local/OAS_file.parquet")
        if not local_file.exists():
            df = get_oas("https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Bashford_2013/csv/ERR220451_Heavy_Bulk.csv.gz")
            df.write_parquet(local_file)
        else:
            df = pl.read_parquet(local_file)
        print(df.head(5))
        ```

    """
    # the OAS has a weird dataformat where the first row is a header
    if isinstance(path, Path):
        path = str(path)

    if not path.endswith(".gz"):
        raise ValueError("The OAS file must be a gzipped CSV file ending with .gz")

    if _is_url(path):
        # For URLs, download to a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=Path(path).suffix, delete=False
        ) as tmp_file:
            _download_file(path, tmp_file.name)
            file_path = Path(tmp_file.name)
    else:
        # For local files, use the path directly
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

    try:
        # lets get only the first 1 row as it has a dict json structure
        # Assume all files are gzipped
        with gzip.open(file_path, "rt") as f:
            first_line_str = f.readline().strip()

            if not first_line_str:
                raise ValueError(f"File {file_path} is empty or has no valid data.")
            try:
                cleaned = first_line_str.strip('"').replace('""', '"')
                first_line_json = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse JSON from the first line of {file_path}: {e}"
                )
        df = pl.scan_csv(
            file_path,
            has_header=True,
            skip_rows=1,  # skip the first row which is a dict
            separator=",",
            infer_schema_length=10000,
        )
        # add metadata from the first line
        for key, value in first_line_json.items():
            if isinstance(value, str):
                value = value.strip('"')
            df = df.with_columns(pl.lit(value).alias(key))
        return df.collect()
    finally:
        # Clean up temporary file if we downloaded one
        if _is_url(path) and file_path.exists():
            file_path.unlink()


if __name__ == "__main__":
    # Example usage
    p = "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Bashford_2013/csv/ERR220451_Heavy_Bulk.csv.gz"
    df = get_oas(p)
    print(df.head(5))
