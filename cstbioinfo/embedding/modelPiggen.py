from typing import List, Literal

import torch
import transformers
from tqdm import tqdm

from .types import Embedder
from .utils import get_device


class pIgGenEmbedder(Embedder):
    dimension: int = 768
    model_max_length: int = 1024

    def __init__(
        self,
        model_name: Literal[
            "ollieturnbull/p-IgGen", "ollieturnbull/p-IgGen-developable"
        ] = "ollieturnbull/p-IgGen",
        device: str | torch.device | None = None,
        cache_dir: str | None = None,
        max_length: int | None = 1024,
    ):
        self.device = get_device(device)

        if isinstance(max_length, int) and max_length > self.model_max_length:
            raise ValueError(
                f"max_length must be less than or equal to {self.model_max_length}, got {max_length}."
            )
        self.max_length = max_length

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.to(self.device)

        # Update dimension based on actual model config
        if hasattr(self.model.config, "hidden_size"):
            self.dimension = self.model.config.hidden_size
        elif hasattr(self.model.config, "n_embd"):
            self.dimension = self.model.config.n_embd

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

            # Tokenize sequences
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]  # [B, L, H] - last layer

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
            del (
                batch,
                input_ids,
                attention_mask,
                outputs,
                hidden_states,
                batch_embeddings,
            )

        return embeddings
