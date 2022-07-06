from typing import Tuple
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np


def get_linkage_matrix(vectors: np.ndarray,
                       dendrogram_title: str,
                       dendrogram_filepath: str,
                       dendrogram_figsize: Tuple[int, int] = (25, 15),
                       metric: str = "cosine",
                       linkage_method: str = "ward") -> np.ndarray:
    condensed_distance_matrix = pdist(vectors, metric=metric)
    Z = linkage(condensed_distance_matrix, method=linkage_method, metric=metric)

    plt.figure(figsize=dendrogram_figsize)
    plt.title(dendrogram_title)
    dendrogram(Z)
    plt.savefig(dendrogram_filepath)

    return Z