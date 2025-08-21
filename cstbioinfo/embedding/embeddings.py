from enum import Enum

from cstbioinfo.embedding.types import Embedder, PairedEmbedder

from .modelAnarcii import ANARCIIEmbedder
from .modelAntiberta2 import AntiBERTa2Embedder
from .modelEsm2 import ESM2Embedder
from .modelIgbert import IgBertEmbedder
from .modelPiggen import pIgGenEmbedder


class EmbedderModel(Enum):
    ANARCII = "anarcii"
    ANTIBERTA2 = "antiberta2"
    ESM2 = "esm2"
    PIGGEN = "piggen"

    @classmethod
    def get_embedder(cls, model: "EmbedderModel", **kwargs) -> Embedder:
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
    IGBERT = "igbert"

    @classmethod
    def get_embedder(cls, model: "PairedEmbedderModel", **kwargs) -> PairedEmbedder:
        if model == cls.IGBERT:
            return IgBertEmbedder(**kwargs)
        else:
            raise ValueError(f"Unknown paired embedder model: {model}")
