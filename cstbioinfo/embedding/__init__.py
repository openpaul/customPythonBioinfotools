"""
Embedding models for protein sequences.

This module provides a unified interface for various protein language models (PLMs)
used for embedding biological sequences like antibodies, TCRs, and general proteins.
It supports both single-sequence embedders and paired-sequence embedders for
antibody heavy/light chain pairs.

The module includes implementations for:
- ANARCII: Antibody numbering and embedding
- AntiBERTa2: Antibody-specific BERT model
- ESM2: Meta's evolutionary scale modeling
- pIgGen: Probabilistic immunoglobulin generator
- IgBert: Paired antibody chain embedder

Examples:
    Basic single sequence embedding:

    >>> from cstbioinfo.embedding.embeddings import EmbedderModel
    >>> # Get an embedder instance
    >>> embedder = EmbedderModel.get_embedder(EmbedderModel.PIGGEN, device="cpu")
    >>> # Embed some sequences
    >>> sequences = ["CASSLGTGQYF", "CASSPGQYF", "CASSQETQYF"]
    >>> embeddings = embedder.embed(sequences, pool="mean", batch_size=8)
    >>> embeddings.shape  # doctest: +SKIP
    torch.Size([3, 768])

    Paired sequence embedding for antibodies:

    >>> from cstbioinfo.embedding.embeddings import PairedEmbedderModel
    >>> # Get a paired embedder
    >>> paired_embedder = PairedEmbedderModel.get_embedder(PairedEmbedderModel.IGBERT, device="cpu")  # doctest: +SKIP
    >>> # Embed heavy/light chain pairs (one can be None)
    >>> paired_seqs = [("QVQLVESGGGLVQPGGSLRLSCAASGFTFS", "DIQMTQSPSSLSASVGDRVTITC")]  # doctest: +SKIP
    >>> paired_embeddings = paired_embedder.embed(paired_seqs, pool="mean")  # doctest: +SKIP
    >>> paired_embeddings.shape  # doctest: +SKIP
    torch.Size([1, 1024])
"""

from .embeddings import EmbedderModel, PairedEmbedderModel
from .modelIgbert import IgBertEmbedder
from .types import Embedder, PairedEmbedder

__all__ = [
    "EmbedderModel",
    "PairedEmbedderModel",
    "Embedder",
    "PairedEmbedder",
    "IgBertEmbedder",
]
