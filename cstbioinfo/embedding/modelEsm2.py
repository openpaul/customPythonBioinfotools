from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .types import Embedder
from .utils import get_device


class ESM2Embedder(Embedder):
    dimension: int = 1280  # ESM2 650M hidden dimension
    model_max_length: int = 1024  # Default max length for ESM2

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str | torch.device | None = None,
        cache_dir: str | None = None,
        max_length: int | None = 1024,
        local_only: bool = False,
    ):
        self.device = get_device(device)
        if isinstance(max_length, int) and max_length > self.model_max_length:
            raise ValueError(
                f"max_length must be less than or equal to {self.max_length}, got {max_length}."
            )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_only
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_only
        ).to(self.device)
        self.model.eval()

    def embed(
        self, sequences: List[str], pool: str = "mean", batch_size: int = 32, **kwargs
    ) -> torch.Tensor:
        """
        Embed a batch of sequences. Returns [batch, hidden_dim] fixed-length embeddings.
        pool: "mean" (default) or "max" pooling over sequence length.
        batch_size: Number of sequences to process at once.
        """
        n_seqs = len(sequences)
        embeddings = torch.empty((n_seqs, self.dimension), device=self.device)

        if self.max_length is None:
            # set to max sequence length
            self.max_length = max(len(seq) for seq in sequences)

        for start in tqdm(range(0, n_seqs, batch_size), desc="Embedding sequences"):
            end = min(start + batch_size, n_seqs)
            batch = sequences[start:end]

            # Encode sequences with explicit parameters for better control
            x = torch.tensor(
                [
                    self.tokenizer.encode(
                        seq,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_special_tokens_mask=True,
                    )
                    for seq in batch
                ]
            ).to(self.device)

            # Create attention mask to ignore padding tokens in pooling
            attention_mask = (x != self.tokenizer.pad_token_id).float().to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    x, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # [B, L, H]

                # Pool over sequence length, ignoring padding tokens
                batch_embeddings = []
                for i, mask in enumerate(attention_mask):
                    valid_tokens = hidden_states[i][mask == 1, :]  # Remove padding

                    if pool == "mean":
                        pooled = valid_tokens.mean(0)
                    elif pool == "max":
                        pooled = valid_tokens.max(0).values
                    else:
                        raise ValueError(f"Unknown pooling method: {pool}")

                    batch_embeddings.append(pooled)

                embeddings[start:end] = torch.stack(batch_embeddings)

            # Clear cache to free memory
            torch.cuda.empty_cache()
            del batch, x, attention_mask, outputs, hidden_states, batch_embeddings

        return embeddings
