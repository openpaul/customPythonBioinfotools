from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pymsaviz import MsaViz
import tempfile
import os
from typing import Optional, Union
import subprocess
import shutil


def _nuc_or_aa(seq: str) -> str:
    """
    Determine if the sequence is nucleotide or amino acid based on content.
    Returns 'nucleotide' or 'amino_acid'.
    """
    if all(base in "ACGTU" for base in seq.upper()):
        return "nucleotide"
    elif all(base in "ACDEFGHIKLMNPQRSTVWY" for base in seq.upper()):
        return "amino_acid"
    else:
        raise ValueError(
            "Sequence contains invalid characters for nucleotide or amino acid."
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
    Perform multiple sequence alignment using BioPython.

    Parameters:
    -----------
    sequences : list[str] or list[SeqRecord]
        List of sequences to align (either as strings or SeqRecord objects)
    method : str, default "muscle"
        Alignment method to use. Options: "muscle", "clustalw", "mafft"
    seq_ids : list[str], optional
        List of sequence identifiers. If not provided, will use seq1, seq2, etc.

    Returns:
    --------
    list[SeqRecord]
        List of aligned sequences as SeqRecord objects
    """
    # Convert string sequences to SeqRecord objects if needed
    if isinstance(sequences[0], str):
        if seq_ids is None:
            seq_ids = [f"seq{i + 1}" for i in range(len(sequences))]

        seq_records = []
        for i, seq in enumerate(sequences):
            record = SeqRecord(Seq(seq), id=seq_ids[i], description="")
            seq_records.append(record)
    else:
        seq_records = sequences

    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as input_file:
        SeqIO.write(seq_records, input_file.name, "fasta")
        input_filename = input_file.name

    sequence_type = _nuc_or_aa(seq_records[0].seq)

    # use mafft for now for alignment
    if method.lower() == "mafft":
        aligned_records = _mafft_alignment(seq_records)
    else:
        raise ValueError(f"Unsupported alignment method: {method}")

    # Clean up temporary file after alignment
    if os.path.exists(input_filename):
        os.unlink(input_filename)
    return aligned_records


def plot_msa(
    sequences: Union[list[str], list[SeqRecord]],
    title: str = "Multiple Sequence Alignment",
    seq_ids: Optional[list[str]] = None,
    **kwargs,
) -> MsaViz:
    aligned_sequences = msa(sequences, seq_ids=seq_ids)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta") as input_file:
        SeqIO.write(aligned_sequences, input_file.name, "fasta")
        input_filename = input_file.name

        mv = MsaViz(input_filename, **kwargs)
        return mv
