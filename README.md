# cstbioinfo

My personal bioinfo useful tools package.
You can use it but changes to the tools will appear with minimal versioning.

If you want a reproducible experience make sure to pin the version in your dependency.

## Features

### MSA Module
Multiple Sequence Alignment tools with visualization support:
- **Alignment**: Uses BioPython for sequence alignment
- **Visualization**: Integrates with pyMSAviz for beautiful MSA plots
- **Notebook-friendly**: Optimized for Jupyter notebook workflows

### Taxonomy Module
Taxonomic data handling and analysis tools:
- **Tax ID resolution**: Convert between different taxonomic identifiers
- **Lineage analysis**: Work with taxonomic hierarchies
- **Data integration**: Compatible with common taxonomic databases

### Embedding Module *(requires `[embedding]` or `[full]` install)*
Protein sequence embedding using state-of-the-art models:
- **ESM2**: Facebook's ESM-2 transformer model for protein embeddings
- **ANARCII**: Specialized embeddings for antibodies and T-cell receptors
- **Batch processing**: Efficient handling of large sequence sets
- **Flexible pooling**: Mean or max pooling strategies

## Installation

This package supports multiple installation modes depending on your needs:

### Basic Installation (Taxonomy & MSA only)
For basic bioinformatics work with taxonomies and multiple sequence alignments:
```bash
pip install cstbioinfo
# or
uv pip install cstbioinfo
```

### Embedding Mode (ML/AI capabilities)
Adds protein embedding support with ESM2 and ANARCII models:
```bash
pip install "cstbioinfo[embedding]"
# or
uv pip install "cstbioinfo[embedding]"
```

### Full Mode (All features)
Complete functionality for production use:
```bash
pip install "cstbioinfo[full]"
# or
uv pip install "cstbioinfo[full]"
```

### Development Mode
For development, testing, and contributions:
```bash
pip install "cstbioinfo[dev]"
# or
uv pip install "cstbioinfo[dev]"
```

**Local development installation:**
```bash
git clone <repository-url>
cd cstbioinfo
pip install -e ".[dev]"
```

## Usage

### MSA Alignment and Visualization

```python
from cstbioinfo.MSA import align_and_plot

# Your sequences
sequences = [
    "ACGTACGTACGTACGT",
    "ACGTACGTACGAACGT", 
    "ACGTACGTACGGACGT"
]

# Align and visualize in one step
mv = align_and_plot(
    sequences,
    seq_ids=["seq1", "seq2", "seq3"],
    title="My MSA",
    color_scheme="Clustal"
)

# Display in notebook
mv.show()
```

### Protein Embeddings *(requires `[embedding]` install)*

```python
from cstbioinfo.embedding import ESM2Embedder, ANARCIIEmbedder

# ESM2 embeddings for general proteins
esm2 = ESM2Embedder(device="cuda")  # or "cpu"
sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
embeddings = esm2.embed(sequences, pool="mean", batch_size=16)
print(f"Embedding shape: {embeddings.shape}")  # [batch_size, 1280]

# ANARCII embeddings for antibodies/TCRs
anarcii = ANARCIIEmbedder(model_type="antibody", device="cuda")
ab_sequences = ["QVQLVESGGGLVQPGGS...", "QVQLVESGGGLVQPGGS..."]
ab_embeddings = anarcii.embed(ab_sequences, pool="mean")
print(f"Antibody embedding shape: {ab_embeddings.shape}")
```

### Individual Functions

```python
from cstbioinfo.MSA import msa, plot_msa

# Step 1: Align sequences
aligned_seqs = msa(sequences, seq_ids=["seq1", "seq2", "seq3"])

# Step 2: Create visualization
mv = plot_msa(
    aligned_seqs, 
    title="Multiple Sequence Alignment",
    width=12,
    show_consensus=True
)

mv.show()
```

### Advanced Options

- **Color schemes**: "Clustal", "Zappo", "Taylor"
- **Wrapping**: Set `wrap_length` for long sequences  
- **Position range**: Use `start_pos` and `end_pos`
- **Display**: Toggle consensus, position numbers, etc.

See `examples/msa_example.ipynb` for detailed examples.

## Development

To contribute to this tool, first checkout the code. Then create a new virtual environment:

```bash
git clone <repository-url>
cd cstbioinfo
uv venv
source ./venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

Install in development mode with all dependencies:

```bash
uv pip install -e ".[dev]"
```

This installs the package in editable mode with all optional dependencies plus development tools like `pytest`, `ruff`, `black`, and `mypy`.

Before any PR ensure the tests are passing:

```bash
pytest .
```

## Installation Modes Summary

| Mode | Command | Includes |
|------|---------|----------|
| **Basic** | `pip install cstbioinfo` | MSA, Taxonomy tools |
| **Embedding** | `pip install "cstbioinfo[embedding]"` | Basic + ESM2, ANARCII |
| **Full** | `pip install "cstbioinfo[full]"` | All features |
| **Dev** | `pip install "cstbioinfo[dev]"` | Full + testing tools |
