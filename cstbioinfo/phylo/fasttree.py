import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Union

from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Phylo import Newick
from cstbioinfo.msa import _nuc_or_aa, msa


class _FastTree:
    """
    Wrapper for FastTree to compute phylogenetic trees from aligned sequences.
    """

    def __init__(self, fasttree_path: str = "fasttree"):
        self.fasttree_path = fasttree_path
        if not shutil.which(self.fasttree_path):
            raise ValueError(f"FastTree executable not found at {self.fasttree_path}")

    def compute_tree(
        self,
        aligned_sequences: Union[str, List[SeqRecord]],
    ) -> Newick.Tree:
        # simple subproecss call to fasttree
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "aligned.fasta"
            output_path = Path(tmpdir) / "tree.nwk"

            # write aligned sequences to input_path
            if isinstance(aligned_sequences, str):
                with open(input_path, "w") as f:
                    f.write(aligned_sequences)
            else:
                SeqIO.write(aligned_sequences, input_path, "fasta")

            cmd = [self.fasttree_path]
            first_seq = (
                (
                    str(aligned_sequences[0].seq)
                    if isinstance(aligned_sequences[0], SeqRecord)
                    else aligned_sequences[0]
                )
                .replace("-", "")
                .replace(" ", "")
            )
            if _nuc_or_aa(first_seq) == "nuc":
                cmd.append("-nt")
            cmd += [str(input_path)]

            with open(output_path, "w") as out_f:
                subprocess.run(cmd, stdout=out_f, check=True)
            tree = Phylo.read(output_path, "newick")  # pyright: ignore[reportPrivateImportUsage]
            return tree


def compute_tree_fast(
    sequences: Union[list[str], list[SeqRecord]],
    fasttree_path: str = "fasttree",
) -> Newick.Tree:
    """
    Compute a phylogenetic tree using FastTree from a list of sequences.

    Args:
        sequences: List of sequences as strings or SeqRecord objects.
        fasttree_path: Path to the FastTree executable.
    Returns:
        A Bio.Phylo tree object representing the phylogenetic tree.

    Example:
        ```python
        from Bio import SeqIO
        from cstbioinfo.phylo.fasttree import compute_tree_fast
        sequences = list(SeqIO.parse("sequences.fasta", "fasta"))
        tree = compute_tree_fast(sequences, fasttree_path="/path/to/fasttree")
        ```
    """
    aligned = msa(sequences)

    # run fasttree
    ft = _FastTree(fasttree_path=fasttree_path)
    tree = ft.compute_tree(aligned)
    return tree
