"""
Clustering Algorithms
K-Means and baseline clustering implementations
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def perform_kmeans_clustering(X, n_clusters=2, random_state=42):
    """
    Perform K-Means clustering.

    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        clusters: Cluster assignments
        model: Fitted K-Means model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)

    print(f"K-Means clustering completed with {n_clusters} clusters")
    print(f"Cluster distribution: {np.bincount(clusters)}")

    return clusters, kmeans


def baseline_pca_clustering(X, n_components=16, n_clusters=2, random_state=42):
    """
    Baseline method: PCA + K-Means clustering.

    Args:
        X: Feature matrix
        n_components: Number of PCA components
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        pca_features: PCA-transformed features
        clusters: Cluster assignments
        pca_model: Fitted PCA model
        kmeans_model: Fitted K-Means model
    """
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA completed: {n_components} components, "
          f"explained variance: {explained_var:.4f}")

    # Cluster PCA features
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(pca_features)

    print(f"K-Means clustering on PCA features completed")
    print(f"Cluster distribution: {np.bincount(clusters)}")

    return pca_features, clusters, pca, kmeans


def get_cluster_statistics(clusters, labels=None):
    """
    Compute cluster statistics.

    Args:
        clusters: Cluster assignments
        labels: True labels (optional)

    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_clusters': len(np.unique(clusters)),
        'cluster_sizes': np.bincount(clusters).tolist(),
        'cluster_distribution': dict(zip(*np.unique(clusters, return_counts=True)))
    }

    if labels is not None:
        # Compute purity for each cluster
        unique_clusters = np.unique(clusters)
        purities = []

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_labels = labels[mask]

            if len(cluster_labels) > 0:
                # Most common label
                unique, counts = np.unique(cluster_labels, return_counts=True)
                purity = counts.max() / len(cluster_labels)
                purities.append(purity)

        stats['average_purity'] = np.mean(purities)

    return stats