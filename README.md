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

## Installation

Install this package using `pip` or `uv`:

```bash
uv pip install cstbioinfo
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
cd cstbioinfo
uv venv
source ./venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
uv pip install -e '.[test]'
```

Before any PR ensure the tests are passing with:

```bash
pytest .
```
