from typing import List, Tuple
from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer

from .types import PairedEmbedder
from .utils import get_device


class IgBertEmbedder(PairedEmbedder):
    dimension: int = 1024  # BERT base hidden dimension
    model_max_length: int = 512  # Default max length for BERT models

    def __init__(
        self,
        model_name: str = "Exscientia/IgBert",
        device: str | torch.device | None = None,
        cache_dir: str | None = None,
        max_length: int | None = 512,
    ):
        self.device = get_device(device)
        if isinstance(max_length, int) and max_length > self.model_max_length:
            raise ValueError(
                f"max_length must be less than or equal to {self.model_max_length}, got {max_length}."
            )
        self.max_length = max_length or self.model_max_length

        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False, cache_dir=cache_dir
        )
        self.model = BertModel.from_pretrained(
            model_name, add_pooling_layer=False, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

    def embed(
        self,
        sequences: List[Tuple[str | None, str | None]],
        pool: str = "mean",
        batch_size: int = 16,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed a batch of paired sequences. Returns [batch, hidden_dim] fixed-length embeddings.

        Args:
            sequences: List of tuples containing (heavy_chain, light_chain) sequences.
                       Either chain can be None, but at least one must be provided.
            pool: "mean" (default) or "max" pooling over sequence length.
            batch_size: Number of sequences to process at once.

        Returns:
            Tensor of shape [batch_size, hidden_dim] containing sequence embeddings.
        """
        n_seqs = len(sequences)
        embeddings = torch.empty((n_seqs, self.dimension), device=self.device)

        # Prepare paired sequences for tokenizer
        paired_sequences = []
        for heavy_chain, light_chain in sequences:
            # Handle cases where one chain might be None
            if heavy_chain is None and light_chain is None:
                raise ValueError("At least one chain (heavy or light) must be provided")
            elif heavy_chain is None:
                # Only light chain - split string into individual amino acids
                if light_chain is not None:
                    paired_seq = " ".join(list(light_chain))
                else:
                    raise ValueError(
                        "Light chain cannot be None when heavy chain is None"
                    )
            elif light_chain is None:
                # Only heavy chain - split string into individual amino acids
                if heavy_chain is not None:
                    paired_seq = " ".join(list(heavy_chain))
                else:
                    raise ValueError(
                        "Heavy chain cannot be None when light chain is None"
                    )
            else:
                # Both chains - format as heavy [SEP] light
                paired_seq = (
                    " ".join(list(heavy_chain))
                    + " [SEP] "
                    + " ".join(list(light_chain))
                )

            paired_sequences.append(paired_seq)

        for start in tqdm(
            range(0, n_seqs, batch_size), desc="Embedding paired sequences"
        ):
            end = min(start + batch_size, n_seqs)
            batch_sequences = paired_sequences[start:end]

            # Tokenize batch
            tokens = self.tokenizer.batch_encode_plus(
                batch_sequences,
                add_special_tokens=True,
                padding="longest",  # Pad to longest sequence in batch
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )

            # Move tokens to device
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            special_tokens_mask = tokens["special_tokens_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get last hidden states [batch_size, seq_len, hidden_dim]
                residue_embeddings = outputs.last_hidden_state

                # Mask special tokens before pooling (following IgBert documentation)
                residue_embeddings[special_tokens_mask == 1] = 0

                if pool == "mean":
                    # Sum embeddings and divide by sequence lengths (excluding special tokens)
                    sequence_embeddings_sum = residue_embeddings.sum(1)
                    sequence_lengths = torch.sum(
                        special_tokens_mask == 0, dim=1
                    ).float()
                    batch_embeddings = (
                        sequence_embeddings_sum / sequence_lengths.unsqueeze(1)
                    )
                elif pool == "max":
                    # Max pooling over non-special tokens
                    batch_embeddings = []
                    for i, mask in enumerate(special_tokens_mask):
                        # Get embeddings for non-special tokens
                        valid_embeddings = residue_embeddings[i][mask == 0, :]
                        if valid_embeddings.size(0) > 0:
                            pooled = valid_embeddings.max(0).values
                        else:
                            # Fallback if no valid tokens (shouldn't happen with proper input)
                            pooled = torch.zeros(self.dimension, device=self.device)
                        batch_embeddings.append(pooled)
                    batch_embeddings = torch.stack(batch_embeddings)
                else:
                    raise ValueError(f"Unknown pooling method: {pool}")

                embeddings[start:end] = batch_embeddings

            # Clear cache to free memory
            torch.cuda.empty_cache()
            del tokens, input_ids, attention_mask, special_tokens_mask
            del outputs, residue_embeddings

        return embeddings
