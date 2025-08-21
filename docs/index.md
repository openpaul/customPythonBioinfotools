# cstbioinfo

Personal bioinformatics tools for Python.

!!! warning "Development Notice"
    This is a personal toolkit. Pin versions for reproducibility as changes may occur with minimal versioning.

## Features

- **MSA**: Multiple sequence alignment with visualization using pyMSAviz
- **Taxonomy**: Tax ID resolution and lineage analysis with NCBI and UniProt data
- **Embeddings**: ESM2 and ANARCII protein embeddings *(optional)*

## Quick Start

### Installation

```bash
# Basic installation (MSA + taxonomy)
pip install cstbioinfo

# With protein embeddings
pip install "cstbioinfo[embedding]"

# Complete installation
pip install "cstbioinfo[full]"
```

### Basic Usage

```python
from cstbioinfo.tax import TaxId, UniProtTax

# NCBI taxonomy lookup
tax = TaxId(9606)  # Human
print(f"Species: {tax.Species}")
print(f"Genus: {tax.Genus}")

# UniProt taxonomy database
uniprot_tax = UniProtTax()
lineage = uniprot_tax.get_lineage(9606)
print(lineage)
```

## Documentation Structure

- [Taxonomy API](api/tax.md) - Tax ID resolution and lineage analysis
- [MSA API](api/msa.md) - Multiple sequence alignment tools
- [Embeddings API](api/embeddings.md) - Protein embedding models
- [Examples](examples.md) - Detailed usage examples

## GitHub Repository

Visit the [GitHub repository](https://github.com/openpaul/customPythonBioinfotools) for source code, issues, and contributions.
