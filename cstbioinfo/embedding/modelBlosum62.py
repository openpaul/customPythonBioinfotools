from typing import List, Tuple

import torch
from tqdm import tqdm

from .types import Embedder, PairedEmbedder

# BLOSUM62 matrix for the 20 standard amino acids (rows = query, columns = alphabet order)
_ALPHABET = "ARNDCQEGHILKMFPSTWYV"

_BLOSUM62_MATRIX = {
    #          A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    "A": [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    "R": [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    "N": [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    "D": [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    "C": [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "Q": [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    "E": [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    "G": [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    "P": [
        -1,
        -2,
        -2,
        -1,
        -3,
        -1,
        -1,
        -2,
        -2,
        -3,
        -3,
        -1,
        -2,
        -4,
        7,
        -1,
        -1,
        -4,
        -3,
        -2,
    ],
    "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    "V": [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
}

_AA_DIM = len(_ALPHABET)  # 20

# Score used for every position when the character is a stop codon (*)
_STOP_CODON_SCORE: float = -4.0

# Build lookup tensor: index → 20-dim BLOSUM62 row
#   0          : padding / fully unknown residue  → all zeros
#   1 .. 20    : standard amino acids in _ALPHABET order
#   21         : stop codon (*) → all -4
_BLOSUM62_TENSOR: torch.Tensor = torch.zeros(_AA_DIM + 2, _AA_DIM, dtype=torch.float32)
for _i, _aa in enumerate(_ALPHABET):
    _BLOSUM62_TENSOR[_i + 1] = torch.tensor(_BLOSUM62_MATRIX[_aa], dtype=torch.float32)
_BLOSUM62_TENSOR[_AA_DIM + 1] = _STOP_CODON_SCORE  # index 21 → stop codon

# Mapping from amino acid character to integer index (1-based; 0 = unknown/padding)
_AA_TO_IDX: dict[str, int] = {aa: i + 1 for i, aa in enumerate(_ALPHABET)}
_AA_TO_IDX["*"] = _AA_DIM + 1  # stop codon → index 21


def _encode_sequence(seq: str, max_length: int, device: torch.device) -> torch.Tensor:
    """
    Encode a single sequence as a padded BLOSUM62 matrix.

    Each amino acid is looked up in the BLOSUM62 matrix (20-dim row).
    The sequence is truncated or zero-padded to ``max_length`` positions.
    Stop codons (``*``) are encoded as a row of -4. Unknown characters are
    encoded as zero vectors.

    Args:
        seq: Amino acid sequence string.
        max_length: Fixed number of positions to encode.
        device: Target torch device.

    Returns:
        Tensor of shape ``(max_length * _AA_DIM,)`` (flattened).
    """
    indices = [_AA_TO_IDX.get(aa.upper(), 0) for aa in seq[:max_length]]
    # Pad with zeros (index 0) up to max_length
    indices += [0] * (max_length - len(indices))
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    blosum_table = _BLOSUM62_TENSOR.to(device)
    encoded = blosum_table[idx_tensor]  # (max_length, _AA_DIM)
    return encoded.flatten()  # (max_length * _AA_DIM,)


class Blosum62Embedder(Embedder):
    """
    Fixed-length BLOSUM62 embedding for single amino acid sequences.

    Each sequence is truncated or zero-padded to ``max_length`` positions.
    Every position is encoded as its 20-dimensional BLOSUM62 substitution row.
    The result is flattened to a vector of size ``max_length × 20``.

    Unknown or padding positions are represented as zero vectors.

    Args:
        max_length: Number of positions to pad/truncate each sequence to.
                    Default is 150.
        device: Torch device to use. If ``None``, auto-selected.

    Example:
        >>> embedder = Blosum62Embedder(max_length=150, device="cpu")
        >>> embedder.dimension
        3000
        >>> t = embedder.embed(["ACDEF", "RKST"])
        >>> t.shape
        torch.Size([2, 3000])
    """

    def __init__(
        self,
        max_length: int = 150,
        device: str | torch.device | None = None,
    ):
        from .utils import get_device

        self.max_length = max_length
        self.device = get_device(device)
        self.dimension = max_length * _AA_DIM

    def embed(
        self,
        sequences: List[str],
        pool: str = "mean",
        batch_size: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed a list of sequences using BLOSUM62.

        Note: ``pool`` is accepted for API compatibility but has no effect —
        BLOSUM62 encoding is position-based and does not require pooling.

        Args:
            sequences: List of amino acid sequence strings.
            batch_size: Number of sequences processed at once (controls memory use).

        Returns:
            Tensor of shape ``(len(sequences), max_length * 20)``.
        """
        n_seqs = len(sequences)
        embeddings = torch.empty((n_seqs, self.dimension), device=self.device)

        for start in tqdm(range(0, n_seqs, batch_size), desc="Embedding sequences"):
            end = min(start + batch_size, n_seqs)
            batch = sequences[start:end]

            batch_enc = torch.stack(
                [_encode_sequence(seq, self.max_length, self.device) for seq in batch]
            )
            embeddings[start:end] = batch_enc

        return embeddings


class Blosum62PairedEmbedder(PairedEmbedder):
    """
    Fixed-length BLOSUM62 embedding for paired amino acid sequences (e.g. heavy/light chains).

    Each chain is independently truncated or zero-padded to ``max_length`` positions
    and encoded via the BLOSUM62 substitution matrix (20 dimensions per position).
    The two chain embeddings are concatenated, giving a final vector of size
    ``2 × max_length × 20``.

    If one chain is ``None``, its slot is filled with zeros.

    Args:
        max_length: Number of positions per chain. Default is 150.
        device: Torch device to use. If ``None``, auto-selected.

    Example:
        >>> embedder = Blosum62PairedEmbedder(max_length=150, device="cpu")
        >>> embedder.dimension
        6000
        >>> seqs = [("QVQLVESG", "DIQMTQSP"), ("EVQLVESG", None)]
        >>> t = embedder.embed(seqs)
        >>> t.shape
        torch.Size([2, 6000])
    """

    def __init__(
        self,
        max_length: int = 150,
        device: str | torch.device | None = None,
    ):
        from .utils import get_device

        self.max_length = max_length
        self.device = get_device(device)
        self.dimension = 2 * max_length * _AA_DIM

    def embed(
        self,
        sequences: List[Tuple[str | None, str | None]],
        pool: str = "mean",
        batch_size: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed a list of paired sequences using BLOSUM62.

        Note: ``pool`` is accepted for API compatibility but has no effect —
        BLOSUM62 encoding is position-based and does not require pooling.

        Args:
            sequences: List of ``(chain_1, chain_2)`` tuples. Either chain may be
                       ``None`` (its slot is zero-padded). Both chains cannot be ``None``.
            batch_size: Number of sequence pairs processed at once.

        Returns:
            Tensor of shape ``(len(sequences), 2 * max_length * 20)``.
        """
        n_seqs = len(sequences)
        embeddings = torch.empty((n_seqs, self.dimension), device=self.device)
        half_dim = self.max_length * _AA_DIM

        for start in tqdm(
            range(0, n_seqs, batch_size), desc="Embedding paired sequences"
        ):
            end = min(start + batch_size, n_seqs)
            batch = sequences[start:end]

            batch_enc = []
            for chain_1, chain_2 in batch:
                if chain_1 is None and chain_2 is None:
                    raise ValueError("At least one chain must be provided per pair.")

                enc_1 = (
                    _encode_sequence(chain_1, self.max_length, self.device)
                    if chain_1 is not None
                    else torch.zeros(half_dim, device=self.device)
                )
                enc_2 = (
                    _encode_sequence(chain_2, self.max_length, self.device)
                    if chain_2 is not None
                    else torch.zeros(half_dim, device=self.device)
                )
                batch_enc.append(torch.cat([enc_1, enc_2]))

            embeddings[start:end] = torch.stack(batch_enc)

        return embeddings
