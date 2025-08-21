from enum import Enum

from cstbioinfo.embedding.types import Embedder

from .modelAnarcii import ANARCIIEmbedder
from .modelEsm2 import ESM2Embedder

# enum of all embedders named models with members ANARCII and ESM2


class EmbedderModel(Enum):
    ANARCII = "anarcii"
    ESM2 = "esm2"

    @classmethod
    def get_embedder(cls, model: "EmbedderModel", **kwargs) -> Embedder:
        if model == cls.ANARCII:
            return ANARCIIEmbedder(**kwargs)
        elif model == cls.ESM2:
            return ESM2Embedder(**kwargs)
        else:
            raise ValueError(f"Unknown embedder model: {model}")
