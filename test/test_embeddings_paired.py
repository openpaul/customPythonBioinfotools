from cstbioinfo.embedding.embeddings import PairedEmbedderModel


def test_embedding_igbert():
    model = PairedEmbedderModel.IGBERT
    embedder = PairedEmbedderModel.get_embedder(model, device="cpu")
    heavy_seqs = ["CASSLGTGQYF", None, "CASSLGTGQYF"]
    light_seqs = ["CASSLGTGQYF", "CASSLGTGQYF", None]
    paired_sequences = list(zip(heavy_seqs, light_seqs))
    embeddings = embedder.embed(paired_sequences, pool="mean", batch_size=1)
    assert embeddings.shape == (3, embedder.dimension), "Embedding shape mismatch"
