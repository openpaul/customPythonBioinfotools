from typing import List

import numpy as np
import torch
from anarcii.inference.model_loader import Loader
from anarcii.input_data_processing.tokeniser import NumberingTokeniser

from cstbioinfo.embedding.types import Embedder


class ANARCIIEmbedder(Embedder):
    dimension: int = 128

    def __init__(
        self,
        model_type: str = "tcr",
        mode: str = "accuracy",
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.device = device
        self.model_type = model_type.lower()
        self.mode = mode

        # Validate model_type
        if self.model_type not in ["antibody", "shark", "tcr"]:
            raise ValueError(
                f"Unknown model_type: {model_type}. Must be one of: antibody, shark, tcr"
            )

        # Initialize tokenizer based on model type
        if self.model_type in ["antibody", "shark"]:
            self.tokeniser = NumberingTokeniser("protein_antibody")
        elif self.model_type == "tcr":
            self.tokeniser = NumberingTokeniser("protein_tcr")

        # Load model
        loader = Loader(self.model_type, self.mode, self.device)
        self.model = loader.model
        self.model.eval()

    def _tokenize_batch(self, sequences: List[str]) -> torch.Tensor:
        # Add start/end tokens and pad to max length
        seqs = [
            [self.tokeniser.start] + list(seq) + [self.tokeniser.end]
            for seq in sequences
        ]
        max_len = max(len(s) for s in seqs)
        _pad_idx = self.tokeniser.char_to_int[self.tokeniser.pad]
        batch = [s + [self.tokeniser.pad] * (max_len - len(s)) for s in seqs]
        tokenized = np.array([self.tokeniser.encode(s) for s in batch])

        return torch.tensor(tokenized, dtype=torch.long, device=self.device)  # [B, L]

    def embed(
        self, sequences: List[str], pool: str = "mean", batch_size: int = 32
    ) -> torch.Tensor:
        """
        Embed a batch of sequences. Returns [batch, hidden_dim] fixed-length embeddings.
        pool: "mean" (default) or "max" pooling over sequence length.
        batch_size: Number of sequences to process at once.
        """
        # Validate sequences
        if any("*" in seq for seq in sequences):
            raise ValueError("Sequences must not contain stop codons ('*').")

        n_seqs = len(sequences)
        embeddings = torch.empty((n_seqs, self.dimension), device=self.device)

        for start in range(0, n_seqs, batch_size):
            end = min(start + batch_size, n_seqs)
            batch = sequences[start:end]

            with torch.no_grad():
                tokens = self._tokenize_batch(batch)  # [B, L]
                src_mask = self.model.make_src_mask(tokens)
                enc = self.model.encoder(tokens, src_mask)  # [B, L, H]

                # Create padding mask for proper pooling
                pad_mask = (
                    tokens != self.tokeniser.char_to_int[self.tokeniser.pad]
                )  # [B, L]

                # Pool over sequence length, ignoring padding tokens
                if pool == "mean":
                    # Mean over non-pad positions
                    sum_enc = (enc * pad_mask.unsqueeze(-1)).sum(dim=1)
                    lengths = pad_mask.sum(dim=1).clamp(min=1)
                    pooled = sum_enc / lengths.unsqueeze(-1)
                elif pool == "max":
                    # Max over non-pad positions
                    enc_masked = enc.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
                    pooled = enc_masked.max(dim=1).values
                else:
                    raise ValueError(
                        f"Unknown pooling method: {pool}. Must be 'mean' or 'max'"
                    )

                embeddings[start:end] = pooled

            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del tokens, src_mask, enc, pad_mask, pooled

        return embeddings
