"""
Multiple Sequence Alignment (MSA) module for sequence alignment and visualization.

This module provides tools for performing multiple sequence alignments using
external alignment tools (currently MAFFT) and visualizing the results using
pymsaviz for publication-quality alignment plots.

Examples:
    Basic alignment and plotting:

    ```python
    from cstbioinfo.msa import msa, plot_msa

    # List of protein sequences
    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVNAL",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVSAL"
    ]

    # Perform alignment
    aligned = msa(sequences, method="mafft")

    # Visualize alignment
    viz = plot_msa(aligned, title="My Protein Alignment")
    viz.show()
    ```

    Working with FASTA files:

    ```python
    from Bio import SeqIO
    from cstbioinfo.msa import msa, plot_msa

    # Read sequences from FASTA file
    sequences = list(SeqIO.parse("sequences.fasta", "fasta"))

    # Align and visualize
    aligned = msa(sequences)
    viz = plot_msa(aligned)
    viz.savefig("alignment.png")
    ```

Requirements:
    - MAFFT must be installed and available in PATH for sequence alignment
    - pymsaviz for visualization functionality
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Union

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pymsaviz import MsaViz


def _nuc_or_aa(seq: str | SeqRecord) -> str:
    """
    Determine if the sequence is nucleotide or amino acid based on content.
    Returns 'nucleotide' or 'amino_acid'.
    """
    if isinstance(seq, SeqRecord):
        seq = str(seq.seq)
    if all(base in "ACGTU" for base in seq.upper()):
        return "nucleotide"
    elif all(base in "ACDEFGHIKLMNPQRSTVWY" for base in seq.upper()):
        return "amino_acid"
    else:
        invalid_chars = set(seq.upper()) - set("ACGTU") - set("ACDEFGHIKLMNPQRSTVWY")
        raise ValueError(
            f"Sequence contains invalid characters: {invalid_chars}. "
            "Cannot determine if nucleotide or amino acid."
        )


def _mafft_alignment(sequences: list[SeqRecord]) -> list[SeqRecord]:
    if shutil.which("mafft") is None:
        raise RuntimeError("MAFFT is not installed or not found in PATH.")
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "input.fasta")
        output_file = os.path.join(temp_dir, "output.fasta")

        SeqIO.write(sequences, input_file, "fasta")

        # Run MAFFT alignment
        subprocess.run(
            ["mafft", "--auto", input_file],
            stdout=open(output_file, "w"),
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return list(SeqIO.parse(output_file, "fasta"))


def msa(
    sequences: Union[list[str], list[SeqRecord]],
    method: str = "mafft",
    seq_ids: Optional[list[str]] = None,
) -> list[SeqRecord]:
    """
    Perform multiple sequence alignment using external alignment tools.

    Aligns a set of sequences using the specified alignment method. Currently
    supports MAFFT alignment with automatic detection of sequence type
    (nucleotide vs amino acid).

    Args:
        sequences: List of sequences to align (either as strings or SeqRecord objects)
        method: Alignment method to use. Currently only "mafft" is supported
        seq_ids: List of sequence identifiers. If not provided, will use seq1, seq2, etc.

    Returns:
        List of aligned sequences as SeqRecord objects

    Raises:
        RuntimeError: If MAFFT is not installed or not found in PATH
        ValueError: If unsupported alignment method is specified

    Examples:
        ```python
        # Align protein sequences
        sequences = [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGAQ"
        ]

        aligned = msa(sequences, method="mafft")
        for record in aligned:
            print(f">{record.id}")
            print(record.seq)
        ```

    Note:
        Requires MAFFT to be installed and available in PATH.
    """
    # Convert string sequences to SeqRecord objects if needed
    if isinstance(sequences[0], str):
        if seq_ids is None:
            seq_ids = [f"seq{i + 1}" for i in range(len(sequences))]

        seq_records: list[SeqRecord] = []
        for i, seq in enumerate(sequences):
            if isinstance(seq, str):
                record = SeqRecord(Seq(seq), id=seq_ids[i], description="")
                seq_records.append(record)
    else:
        seq_records = sequences  # type: ignore[assignment]

    # Check sequence type (keeping for potential future use)
    first_seq_str = str(seq_records[0].seq)
    _nuc_or_aa(first_seq_str)

    # use mafft for now for alignment
    if method.lower() == "mafft":
        aligned_records = _mafft_alignment(seq_records)
    else:
        raise ValueError(f"Unsupported alignment method: {method}")

    return aligned_records


def plot_msa(
    sequences: Union[list[str], list[SeqRecord]],
    title: str = "Multiple Sequence Alignment",
    seq_ids: Optional[list[str]] = None,
    **kwargs,
) -> MsaViz:
    """
    Create a visualization of a multiple sequence alignment.

    Performs sequence alignment and generates a publication-quality visualization
    using pymsaviz. The alignment is performed automatically using MAFFT before
    visualization.

    Args:
        sequences: List of sequences to align and visualize (strings or SeqRecord objects)
        title: Title for the alignment visualization
        seq_ids: List of sequence identifiers for labeling
        **kwargs: Additional arguments passed to MsaViz for customization

    Returns:
        MsaViz object for further customization and display/saving

    Examples:
        Basic alignment visualization:

        ```python
        sequences = [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"
        ]

        viz = plot_msa(sequences, title="My Protein Alignment")
        viz.show()  # Display in notebook
        viz.savefig("alignment.png")  # Save to file
        ```

        Customizing the visualization:

        ```python
        viz = plot_msa(
            sequences,
            title="Custom Alignment",
            color_scheme="flower",
            show_consensus=True,
            show_count=True
        )
        ```

    Note:
        Requires MAFFT for alignment and pymsaviz for visualization.
        See pymsaviz documentation for available kwargs options.
    """
    sequences_seqio = []
    for i, seq in enumerate(sequences):
        if isinstance(seq, str):
            if seq_ids is None:
                seq_id = f"seq{i + 1}"
            else:
                seq_id = seq_ids[i]
            record = SeqRecord(Seq(seq), id=seq_id, description="")
            sequences_seqio.append(record)
        elif isinstance(seq, SeqRecord):
            sequences_seqio.append(seq)
        else:
            raise ValueError("Sequences must be either strings or SeqRecord objects")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta") as input_file:
        SeqIO.write(sequences_seqio, input_file.name, "fasta")
        input_filename = input_file.name

        mv = MsaViz(input_filename, **kwargs)
        return mv


def distance_matrix(
    sequences: list[str] | list[SeqRecord],
    align: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a pairwise distance matrix for a list of sequences.

    Calculates the pairwise distance between sequences based on the number of
    differing positions. Optionally performs multiple sequence alignment before
    distance calculation.

    Args:
        sequences: List of sequences (strings or SeqRecord objects)
        align: Whether to perform multiple sequence alignment before distance calculation
        normalize: Whether to normalize distances by sequence length (proportion of differences)
    Returns:
        np.ndarray: Pairwise distance matrix of shape (N, N) where N is number of sequences

    Examples:
        Basic distance matrix calculation:
        ```python
        sequences = [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGAQ",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"
        ]
        dist_matrix = distance_matrix(sequences, align=True, normalize=True)
    """
    if align is False:
        # ensure all sequences are same length, else we pad with gaps
        max_len = max(
            len(seq) if isinstance(seq, str) else len(str(seq.seq)) for seq in sequences
        )
        padded_seqs = []
        for i, seq in enumerate(sequences):
            if isinstance(seq, str):
                padded_seq = seq.ljust(max_len, "-")
                record = SeqRecord(Seq(padded_seq), id=f"seq{i + 1}", description="")
                padded_seqs.append(record)
            else:
                padded_seq = str(seq.seq).ljust(max_len, "-")
                record = SeqRecord(
                    Seq(padded_seq), id=seq.id, description=seq.description
                )
                padded_seqs.append(record)
        sequences = padded_seqs
    else:
        sequences = msa(sequences)

    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=float)

    # convert to strings
    sequences = [seq if isinstance(seq, str) else str(seq.seq) for seq in sequences]

    for i in range(n):
        for j in range(i + 1, n):
            seq1 = sequences[i]
            seq2 = sequences[j]
            if len(seq1) != len(seq2):
                raise ValueError(
                    "Sequences must be of equal length for distance calculation"
                )
            differences = sum(
                c1 != c2 for c1, c2 in zip(seq1, seq2) if c1 != "-" and c2 != "-"
            )
            valid_positions = sum(c1 != "-" and c2 != "-" for c1, c2 in zip(seq1, seq2))
            if normalize:
                dist = differences / valid_positions if valid_positions > 0 else 0.0
            else:
                dist = differences
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix
