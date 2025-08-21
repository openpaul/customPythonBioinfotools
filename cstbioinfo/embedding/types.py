from typing import List

import torch


class Embedder:
    dimension: int = 0  # Default dimension, should be overridden by subclasses

    def __init__(self):
        raise NotImplementedError(
            "Embedder is an abstract base class. Please use a subclass like ANARCIIEmbedder."
        )

    def embed(self, sequences: List[str], pool: str = "mean", **kwargs) -> torch.Tensor:
        """
        Embed a batch of sequences. Returns [batch, hidden_dim] fixed-length embeddings.
        pool: "mean" (default) or "max" pooling over sequence length.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
