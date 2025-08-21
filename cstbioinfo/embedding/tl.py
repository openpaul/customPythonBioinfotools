import numpy as np
import polars as pl
import torch
import umap as umap_package


def umap(
    data: torch.Tensor | np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> pl.DataFrame:
    """
    Perform UMAP dimensionality reduction on the input data.

    Parameters:
    - data: Input data to be embedded (numpy array or pandas DataFrame).
    - n_neighbors: Number of neighbors to consider for each point.
    - min_dist: Minimum distance between points in the embedding space.
    - metric: Distance metric to use for the embedding.

    Returns:
    - UMAP embedding of the input data.
    """
    reducer = umap_package.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric
    )
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()  # Convert to numpy array if input is a tensor

    reduced = reducer.fit_transform(data)
    return pl.DataFrame(reduced, schema=[f"dim_{i}" for i in range(reduced.shape[1])])
