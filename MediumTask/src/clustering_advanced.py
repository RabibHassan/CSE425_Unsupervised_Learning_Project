"""
Advanced Clustering Module
K-Means, Agglomerative, and DBSCAN clustering
"""

import numpy as np # type: ignore
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN # type: ignore
from sklearn.metrics import silhouette_score # type: ignore
import warnings
warnings.filterwarnings('ignore')


def perform_kmeans(features, n_clusters=10, random_state=42, verbose=True):
    """
    Perform K-Means clustering.
    
    Args:
        features: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
        verbose: Print information
        
    Returns:
        clusters: Cluster assignments
        model: Fitted K-Means model
        score: Silhouette score
    """
    if verbose:
        print(f"\nPerforming K-Means clustering (k={n_clusters})...")
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    clusters = model.fit_predict(features)
    
    # Compute silhouette score
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(features, clusters)
    else:
        score = -1.0
    
    if verbose:
        print(f"✓ K-Means complete")
        print(f"  Clusters found: {len(np.unique(clusters))}")
        print(f"  Silhouette score: {score:.4f}")
        print(f"  Cluster distribution: {np.bincount(clusters)}")
    
    return clusters, model, score


def perform_agglomerative(features, n_clusters=10, linkage='ward', verbose=True):
    """
    Perform Agglomerative (Hierarchical) clustering.
    
    Args:
        features: Feature matrix
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average')
        verbose: Print information
        
    Returns:
        clusters: Cluster assignments
        model: Fitted Agglomerative model
        score: Silhouette score
    """
    if verbose:
        print(f"\nPerforming Agglomerative clustering (k={n_clusters}, linkage={linkage})...")
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    
    clusters = model.fit_predict(features)
    
    # Compute silhouette score
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(features, clusters)
    else:
        score = -1.0
    
    if verbose:
        print(f"✓ Agglomerative complete")
        print(f"  Clusters found: {len(np.unique(clusters))}")
        print(f"  Silhouette score: {score:.4f}")
        print(f"  Cluster distribution: {np.bincount(clusters)}")
    
    return clusters, model, score


def perform_dbscan(features, eps=0.5, min_samples=5, verbose=True):
    """
    Perform DBSCAN clustering.
    
    Args:
        features: Feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum samples in neighborhood
        verbose: Print information
        
    Returns:
        clusters: Cluster assignments (-1 for noise)
        model: Fitted DBSCAN model
        score: Silhouette score (if >1 cluster)
    """
    if verbose:
        print(f"\nPerforming DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
    
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(features)
    
    # Count clusters (excluding noise points)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    # Compute silhouette score (excluding noise)
    if n_clusters > 1:
        # Remove noise points for scoring
        mask = clusters != -1
        if mask.sum() > 0:
            score = silhouette_score(features[mask], clusters[mask])
        else:
            score = -1.0
    else:
        score = -1.0
    
    if verbose:
        print(f"✓ DBSCAN complete")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        print(f"  Silhouette score: {score:.4f}")
        if n_clusters > 0:
            print(f"  Cluster distribution: {np.bincount(clusters[clusters != -1])}")
    
    return clusters, model, score


def cluster_all_methods(features, n_clusters=10, verbose=True):
    """
    Apply all clustering methods to features.
    
    Args:
        features: Feature matrix
        n_clusters: Target number of clusters (for K-Means, Agglomerative)
        verbose: Print information
        
    Returns:
        results: Dictionary with clustering results
    """
    if verbose:
        print("\n" + "="*70)
        print("CLUSTERING WITH MULTIPLE ALGORITHMS")
        print("="*70)
        print(f"Feature shape: {features.shape}")
        print(f"Target clusters: {n_clusters}")
    
    results = {}
    
    # K-Means
    clusters_km, model_km, score_km = perform_kmeans(
        features, n_clusters=n_clusters, verbose=verbose
    )
    results['kmeans'] = {
        'clusters': clusters_km,
        'model': model_km,
        'silhouette': score_km
    }
    
    # Agglomerative (Ward linkage)
    clusters_agg, model_agg, score_agg = perform_agglomerative(
        features, n_clusters=n_clusters, linkage='ward', verbose=verbose
    )
    results['agglomerative'] = {
        'clusters': clusters_agg,
        'model': model_agg,
        'silhouette': score_agg
    }
    
    # DBSCAN (auto-tune eps)
    # Estimate eps from data
    from sklearn.neighbors import NearestNeighbors # type: ignore
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    eps_estimate = np.percentile(distances[:, -1], 90)
    min_samples_estimate = max(5, int(np.log(len(features)))) 
    
    clusters_db, model_db, score_db = perform_dbscan(
        features, eps=eps_estimate, min_samples=min_samples_estimate, verbose=verbose
    )
    results['dbscan'] = {
        'clusters': clusters_db,
        'model': model_db,
        'silhouette': score_db
    }
    
    if verbose:
        print("\n" + "="*70)
        print("✓ All clustering methods complete!")
        print("="*70)
        print("\nSilhouette Scores Summary:")
        print(f"  K-Means: {score_km:.4f}")
        print(f"  Agglomerative: {score_agg:.4f}")
        print(f"  DBSCAN: {score_db:.4f}")
    
    return results


if __name__ == "__main__":
    # Test clustering
    print("Testing Clustering Algorithms...")
    print("="*70)
    
    # Generate dummy data
    n_samples = 1000
    n_features = 64
    features = np.random.randn(n_samples, n_features)
    
    # Test all methods
    results = cluster_all_methods(features, n_clusters=10, verbose=True)
    
    print("\nClustering Test Complete!")
