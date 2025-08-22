from enum import Enum

from cstbioinfo.embedding.types import Embedder, PairedEmbedder

from .modelAnarcii import ANARCIIEmbedder
from .modelAntiberta2 import AntiBERTa2Embedder
from .modelEsm2 import ESM2Embedder
from .modelIgbert import IgBertEmbedder
from .modelPiggen import pIgGenEmbedder


class EmbedderModel(Enum):
    """
    Enumeration of available single-sequence embedding models.

    This enum provides a convenient way to select and instantiate different
    protein language models for embedding biological sequences.

    Available models:
        ANARCII: Antibody Numbering and Antigen Receptor Classification
                 Supports antibody, shark, and TCR sequences
        ANTIBERTA2: Antibody-specific BERT model with multiple variants
        ESM2: Meta's Evolutionary Scale Modeling v2 for general proteins
        PIGGEN: Probabilistic immunoglobulin generator for TCR sequences

    Examples:
        >>> # List all available models
        >>> list(EmbedderModel)  # doctest: +ELLIPSIS
        [<EmbedderModel.ANARCII: 'anarcii'>, <EmbedderModel.ANTIBERTA2: 'antiberta2'>, ...]

        >>> # Get model by name
        >>> EmbedderModel.PIGGEN.value
        'piggen'

        >>> # Create an embedder instance
        >>> embedder = EmbedderModel.get_embedder(EmbedderModel.PIGGEN, device="cpu")
        >>> embedder.dimension  # doctest: +SKIP
        768
    """

    ANARCII = "anarcii"
    ANTIBERTA2 = "antiberta2"
    ESM2 = "esm2"
    PIGGEN = "piggen"

    @classmethod
    def get_embedder(cls, model: "EmbedderModel", **kwargs) -> Embedder:
        """
        Factory method to create an embedder instance.

        Args:
            model: The EmbedderModel enum value specifying which model to use
            **kwargs: Additional keyword arguments passed to the specific embedder
                     constructor. Common arguments include:
                     - device: str or torch.device, computing device ("cpu", "cuda", etc.)
                     - batch_size: int, batch size for inference
                     - model_type: str, specific model variant (for ANARCII, AntiBERTa2)

        Returns:
            Embedder: An instance of the requested embedder model

        Raises:
            ValueError: If the model type is not recognized

        Examples:
            >>> # Create a PIGGEN embedder for CPU
            >>> embedder = EmbedderModel.get_embedder(EmbedderModel.PIGGEN, device="cpu")
            >>> isinstance(embedder, pIgGenEmbedder)  # doctest: +SKIP
            True

            >>> # Create an ANARCII embedder for antibodies
            >>> anarcii = EmbedderModel.get_embedder(
            ...     EmbedderModel.ANARCII,
            ...     device="cpu",
            ...     model_type="antibody"
            ... )  # doctest: +SKIP
            >>> anarcii.model_type  # doctest: +SKIP
            'antibody'

            >>> # Test invalid model raises ValueError
            >>> class InvalidModel(Enum):
            ...     INVALID = "invalid"
            >>> try:
            ...     EmbedderModel.get_embedder(InvalidModel.INVALID)  # doctest: +SKIP
            ... except ValueError as e:
            ...     print("ValueError raised correctly")  # doctest: +SKIP
        """
        if model == cls.ANARCII:
            return ANARCIIEmbedder(**kwargs)
        elif model == cls.ANTIBERTA2:
            return AntiBERTa2Embedder(**kwargs)
        elif model == cls.ESM2:
            return ESM2Embedder(**kwargs)
        elif model == cls.PIGGEN:
            return pIgGenEmbedder(**kwargs)
        else:
            raise ValueError(f"Unknown embedder model: {model}")


class PairedEmbedderModel(Enum):
    """
    Enumeration of available paired-sequence embedding models.

    This enum provides models specifically designed for paired biological sequences,
    such as antibody heavy/light chain pairs or TCR alpha/beta chain pairs.

    Available models:
        IGBERT: BERT-based model for paired immunoglobulin sequences

    Examples:
        >>> # List available paired models
        >>> list(PairedEmbedderModel)
        [<PairedEmbedderModel.IGBERT: 'igbert'>]

        >>> # Get model value
        >>> PairedEmbedderModel.IGBERT.value
        'igbert'

        >>> # Create a paired embedder
        >>> paired_embedder = PairedEmbedderModel.get_embedder(
        ...     PairedEmbedderModel.IGBERT,
        ...     device="cpu"
        ... )  # doctest: +SKIP
        >>> paired_embedder.dimension  # doctest: +SKIP
        1024
    """

    IGBERT = "igbert"

    @classmethod
    def get_embedder(cls, model: "PairedEmbedderModel", **kwargs) -> PairedEmbedder:
        """
        Factory method to create a paired embedder instance.

        Args:
            model: The PairedEmbedderModel enum value specifying which model to use
            **kwargs: Additional keyword arguments passed to the specific embedder
                     constructor. Common arguments include:
                     - device: str or torch.device, computing device ("cpu", "cuda", etc.)
                     - max_length: int, maximum sequence length
                     - cache_dir: str, directory for model caching

        Returns:
            PairedEmbedder: An instance of the requested paired embedder model

        Raises:
            ValueError: If the model type is not recognized

        Examples:
            >>> # Create an IgBert embedder
            >>> embedder = PairedEmbedderModel.get_embedder(
            ...     PairedEmbedderModel.IGBERT,
            ...     device="cpu",
            ...     max_length=256
            ... )  # doctest: +SKIP
            >>> embedder.max_length  # doctest: +SKIP
            256

            >>> # Embed paired sequences (heavy, light)
            >>> sequences = [
            ...     ("QVQLVESGGGLVQPGG", "DIQMTQSPSSLSASVG"),
            ...     ("QVQLVESGGGLVQPGG", None),  # Only heavy chain
            ...     (None, "DIQMTQSPSSLSASVG")   # Only light chain
            ... ]
            >>> embeddings = embedder.embed(sequences, batch_size=2)  # doctest: +SKIP
            >>> embeddings.shape  # doctest: +SKIP
            torch.Size([3, 1024])
        """
        if model == cls.IGBERT:
            return IgBertEmbedder(**kwargs)
        else:
            raise ValueError(f"Unknown paired embedder model: {model}")
