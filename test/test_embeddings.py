from cstbioinfo.embedding.embeddings import EmbedderModel


def test_embedd_anarcii():
    model = EmbedderModel.ANARCII
    embedder = EmbedderModel.get_embedder(model, device="cpu", model_type="antibody")
    sequences = ["CASSLGTGQYF", "CASSLGTGQYF"]
    embeddings = embedder.embed(sequences, pool="mean", batch_size=1)
    assert embeddings.shape == (2, embedder.dimension), "Embedding shape mismatch"


def test_embed_esm2():
    model = EmbedderModel.ESM2
    embedder = EmbedderModel.get_embedder(model, device="cpu")
    sequences = ["CASSLGTGQYF", "CASSLGTGQYF", "CASSLGTGQYF"]
    embeddings = embedder.embed(sequences, pool="mean", batch_size=1)
    assert embeddings.shape == (3, embedder.dimension), "Embedding shape mismatch"
