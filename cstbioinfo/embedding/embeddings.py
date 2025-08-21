from enum import Enum

from cstbioinfo.embedding.types import Embedder

from .modelAnarcii import ANARCIIEmbedder
from .modelAntiberta2 import AntiBERTa2Embedder
from .modelEsm2 import ESM2Embedder
from .modelPiggen import pIgGenEmbedder

# enum of all embedders named models with members ANARCII and ESM2


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
