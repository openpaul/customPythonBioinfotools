from typing import List, Tuple

import torch


class Embedder:
    dimension: int = 0  # Default dimension, should be overridden by subclasses

    def __init__(self):
        raise NotImplementedError(
            "Embedder is an abstract base class. Please use a subclass like ANARCIIEmbedder."
        )

    def embed(self, sequences: List, pool: str = "mean", **kwargs) -> torch.Tensor:
        """
        Embed a batch of sequences. Returns [batch, hidden_dim] fixed-length embeddings.
        pool: "mean" (default) or "max" pooling over sequence length.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class PairedEmbedder(Embedder):
    """
    Base class for paired sequence embedders.
    This class is intended to be subclassed for specific paired embedding models.
    """

    def embed(
        self,
        sequences: List[Tuple[str | None, str | None]],
        pool: str = "mean",
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed a batch of paired sequences. Returns [batch, hidden_dim] fixed-length embeddings.
        pool: "mean" (default) or "max" pooling over sequence length.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
