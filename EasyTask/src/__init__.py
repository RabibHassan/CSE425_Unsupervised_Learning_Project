"""
VAE Music Clustering Package
Contains modules for VAE-based music clustering
"""

__version__ = "1.0.0"
__author__ = "Moin Mostakim"

from .vae_model import BasicVAE
from .dataset import load_dataset, extract_features, preprocess_features
from .clustering import perform_kmeans_clustering, baseline_pca_clustering
from .evaluation import evaluate_clustering
from .visualization import plot_tsne_clusters

__all__ = [
    'BasicVAE',
    'load_dataset',
    'extract_features',
    'preprocess_features',
    'perform_kmeans_clustering',
    'baseline_pca_clustering',
    'evaluate_clustering',
    'plot_tsne_clusters'
]