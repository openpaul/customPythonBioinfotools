from typing import List, Literal
from tqdm import tqdm
import torch
import transformers

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

    def get_batch_log_likelihoods(
        self, sequences: List[str], batch_size: int = 32
    ) -> List[float]:
        """
        Computes the log likelihood for a batch of sequences.
        This method is included to maintain compatibility with the original pIgGen functionality.

        Args:
            sequences (List[str]): A list of sequences for which to compute the likelihood.
            batch_size (int): The size of each batch for processing.
        Returns:
            likelihoods (List[float]): A list of log likelihoods for each sequence.
        """
        likelihoods = []

        # Special token IDs for pIgGen
        bos_token_id = self.tokenizer.encode("1")[0]
        eos_token_id = self.tokenizer.encode("2")[0]
        pad_token_id = self.tokenizer.pad_token_id
        special_token_ids = [bos_token_id, eos_token_id, pad_token_id]

        # Split sequences into batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i : i + batch_size]

            # Tokenize all sequences once
            inputs = self.tokenizer(
                batch_sequences, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # Compute the likelihood for each sequence in the batch
            for input_id, logit in zip(input_ids, logits):
                # align the logits with the input ids
                # (remove first element of logits and last element of input_ids)
                shift_logits = logit[:-1, :].contiguous()
                shift_labels = input_id[1:].contiguous().long()

                # Create mask to exclude special tokens
                mask = torch.ones(shift_labels.shape, dtype=torch.bool).to(self.device)
                for token_id in special_token_ids:
                    mask = mask & (shift_labels != token_id)

                # Compute the negative log-likelihood using cross_entropy,
                # ignoring masked tokens
                nll = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1))[mask],
                    shift_labels.view(-1)[mask],
                    reduction="mean",
                )
                likelihoods.append(-nll)

        return torch.stack(likelihoods, dim=0).cpu().numpy().tolist()
