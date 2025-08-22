import gzip
import json
import tempfile
from pathlib import Path

import polars as pl
import requests


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
