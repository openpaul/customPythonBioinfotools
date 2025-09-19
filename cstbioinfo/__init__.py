"""
cstbioinfo - Personal bioinformatics tools
==========================================

A collection of useful bioinformatics tools and utilities developed for computational biology
and bioinformatics research.

**Note**: I maintain this package for my own work. Functions that end up being used in publications
will be moved to either the publication-specific code repository or published as a package on PyPi.
This is essentially a testing ground for new ideas and code snippets. So use at your own risk!

Development is hosted on GitHub:
[https://github.com/openpaul/customPythonBioinfotools/](https://github.com/openpaul/customPythonBioinfotools/)

## Overview

**cstbioinfo** provides a suite of tools for:

- **Taxonomy**: Work with NCBI and UniProt taxonomic databases
- **Multiple Sequence Alignment (MSA)**: Visualize and analyze MSAs
- **Protein Embeddings**: Generate and work with protein embeddings from various models
- **Immune Repertoire Analysis**: Analyze antibody and TCR sequences

## Quick Start

### Installation

```sh
pip install git+https://github.com/openpaul/customPythonBioinfotools/
```

Or to install with all optional dependencies:

```sh
git clone https://github.com/openpaul/customPythonBioinfotools/
cd customPythonBioinfotools
pip install  '.[full]'
```

### Basic Usage

#### Taxonomy lookups
```python
from cstbioinfo.tax import TaxId

# Get taxonomic information for humans
human = TaxId(9606)
print(f"Species: {human.Species}")  # Homo sapiens
print(f"Family: {human.Family}")    # Hominidae
```

#### MSA visualization
```python
from cstbioinfo.msa import msa, plot_msa

# Load and plot an MSA
alignment = msa("path/to/alignment.fasta")
plot_msa(alignment)
```

#### Immune repertoire analysis
```python
from cstbioinfo.immune import ruzicka_similarity

# Calculate similarity between antibody repertoires
similarity = ruzicka_similarity(
    df,
    feature_columns=["v_call", "j_call", "junction_aa"],
    count_column="count",
    sample_column="sample_id"
)
```

## Modules

The package is organized into several modules:

- **cstbioinfo.tax**: Taxonomic database interfaces and utilities
- **cstbioinfo.msa**: Multiple sequence alignment tools
- **cstbioinfo.embedding**: Protein embedding models and utilities
- **cstbioinfo.immune**: Immune repertoire analysis tools

## Requirements

- Python ≥ 3.10
- See pyproject.toml for full dependency list

## License

Apache-2.0
"""

__version__ = "0.1.2"
