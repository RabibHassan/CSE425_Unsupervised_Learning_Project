"""
Advanced Evaluation Module
Implements all 6 required clustering metrics
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import ( # type: ignore
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import warnings
warnings.filterwarnings('ignore')


def compute_cluster_purity(clusters, true_labels):
    """
    Compute cluster purity.
    
    Purity = (1/N) * Σ max_j |cluster_k ∩ class_j|
    
    Args:
        clusters: Predicted cluster assignments
        true_labels: True class labels
        
    Returns:
        purity: Cluster purity score
    """
    # Convert to numpy arrays
    clusters = np.array(clusters)
    true_labels = np.array(true_labels)
    
    # Remove noise points (if any from DBSCAN)
    mask = clusters != -1
    clusters = clusters[mask]
    true_labels = true_labels[mask]
    
    if len(clusters) == 0:
        return 0.0
    
    # Get unique clusters and labels
    cluster_ids = np.unique(clusters)
    label_ids = np.unique(true_labels)
    
    # Compute purity
    total_correct = 0
    for cluster_id in cluster_ids:
        # Get samples in this cluster
        cluster_mask = clusters == cluster_id
        cluster_labels = true_labels[cluster_mask]
        
        # Find most common label in this cluster
        if len(cluster_labels) > 0:
            label_counts = np.bincount(
                [np.where(label_ids == label)[0][0] for label in cluster_labels]
            )
            max_count = label_counts.max()
            total_correct += max_count
    
    purity = total_correct / len(clusters)
    return purity


def evaluate_clustering_full(features, clusters, true_labels, verbose=True):
    """
    Compute all 6 clustering metrics.
    
    Args:
        features: Feature matrix
        clusters: Predicted cluster assignments
        true_labels: True class labels (genre)
        verbose: Print results
        
    Returns:
        metrics: Dictionary with all metrics
    """
    # Handle noise points from DBSCAN
    mask = clusters != -1
    features_clean = features[mask]
    clusters_clean = clusters[mask]
    labels_clean = true_labels[mask]
    
    # Check if we have valid clusters
    n_clusters = len(np.unique(clusters_clean))
    
    if n_clusters < 2:
        if verbose:
            print("Warning: Less than 2 clusters found, returning default metrics")
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz_index': 0.0,
            'davies_bouldin_index': 999.0,
            'adjusted_rand_index': 0.0,
            'normalized_mutual_info': 0.0,
            'cluster_purity': 0.0,
            'n_clusters': n_clusters,
            'n_samples': len(clusters_clean)
        }
    
    metrics = {}
    
    # 1. Silhouette Score (higher is better, range [-1, 1])
    try:
        metrics['silhouette_score'] = silhouette_score(features_clean, clusters_clean)
    except:
        metrics['silhouette_score'] = -1.0
    
    # 2. Calinski-Harabasz Index (higher is better)
    try:
        metrics['calinski_harabasz_index'] = calinski_harabasz_score(
            features_clean, clusters_clean
        )
    except:
        metrics['calinski_harabasz_index'] = 0.0
    
    # 3. Davies-Bouldin Index (lower is better)
    try:
        metrics['davies_bouldin_index'] = davies_bouldin_score(
            features_clean, clusters_clean
        )
    except:
        metrics['davies_bouldin_index'] = 999.0
    
    # 4. Adjusted Rand Index (higher is better, range [-1, 1])
    try:
        metrics['adjusted_rand_index'] = adjusted_rand_score(
            labels_clean, clusters_clean
        )
    except:
        metrics['adjusted_rand_index'] = 0.0
    
    # 5. Normalized Mutual Information (higher is better, range [0, 1])
    try:
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(
            labels_clean, clusters_clean
        )
    except:
        metrics['normalized_mutual_info'] = 0.0
    
    # 6. Cluster Purity (higher is better, range [0, 1])
    try:
        metrics['cluster_purity'] = compute_cluster_purity(clusters_clean, labels_clean)
    except:
        metrics['cluster_purity'] = 0.0
    
    # Additional info
    metrics['n_clusters'] = n_clusters
    metrics['n_samples'] = len(clusters_clean)
    metrics['n_noise'] = len(clusters) - len(clusters_clean)
    
    if verbose:
        print("\n" + "="*70)
        print("CLUSTERING EVALUATION METRICS")
        print("="*70)
        print(f"Samples: {metrics['n_samples']} (noise: {metrics['n_noise']})")
        print(f"Clusters: {metrics['n_clusters']}")
        print("\nMetrics:")
        print(f"  1. Silhouette Score:        {metrics['silhouette_score']:.4f} ↑")
        print(f"  2. Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f} ↑")
        print(f"  3. Davies-Bouldin Index:    {metrics['davies_bouldin_index']:.4f} ↓")
        print(f"  4. Adjusted Rand Index:     {metrics['adjusted_rand_index']:.4f} ↑")
        print(f"  5. Normalized Mutual Info:  {metrics['normalized_mutual_info']:.4f} ↑")
        print(f"  6. Cluster Purity:          {metrics['cluster_purity']:.4f} ↑")
        print("="*70)
    
    return metrics


def evaluate_all_experiments(features_dict, clustering_results, labels, verbose=True):
    """
    Evaluate all feature variants with all clustering methods.
    
    Args:
        features_dict: Dictionary of feature variants
        clustering_results: Dictionary of clustering results for each variant
        labels: True genre labels
        verbose: Print progress
        
    Returns:
        results_df: DataFrame with all results
    """
    if verbose:
        print("\n" + "="*70)
        print("EVALUATING ALL EXPERIMENTS")
        print("="*70)
    
    all_results = []
    
    for feature_name, features in features_dict.items():
        if verbose:
            print(f"\nEvaluating: {feature_name}")
            print("-" * 70)
        
        # Get clustering results for this feature variant
        if feature_name not in clustering_results:
            if verbose:
                print(f"  Skipping (no clustering results)")
            continue
        
        cluster_dict = clustering_results[feature_name]
        
        for method_name, method_data in cluster_dict.items():
            if verbose:
                print(f"  Method: {method_name}")
            
            clusters = method_data['clusters']
            
            # Evaluate
            metrics = evaluate_clustering_full(
                features, clusters, labels, verbose=False
            )
            
            # Add identifiers
            metrics['feature_type'] = feature_name
            metrics['clustering_method'] = method_name
            
            all_results.append(metrics)
            
            if verbose:
                print(f"    Silhouette: {metrics['silhouette_score']:.4f}")
                print(f"    CH Index: {metrics['calinski_harabasz_index']:.2f}")
                print(f"    ARI: {metrics['adjusted_rand_index']:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    col_order = [
        'feature_type', 'clustering_method', 
        'silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index',
        'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity',
        'n_clusters', 'n_samples', 'n_noise'
    ]
    results_df = results_df[col_order]
    
    if verbose:
        print("\n" + "="*70)
        print("✓ All experiments evaluated!")
        print("="*70)
    
    return results_df


def find_best_configurations(results_df, verbose=True):
    """
    Find best configurations for each metric.
    
    Args:
        results_df: Results DataFrame
        verbose: Print best configs
        
    Returns:
        best_configs: Dictionary of best configurations
    """
    if verbose:
        print("\n" + "="*70)
        print("BEST CONFIGURATIONS")
        print("="*70)
    
    best_configs = {}
    
    # Metrics where higher is better
    higher_better = [
        'silhouette_score',
        'calinski_harabasz_index',
        'adjusted_rand_index',
        'normalized_mutual_info',
        'cluster_purity'
    ]
    
    # Metric where lower is better
    lower_better = ['davies_bouldin_index']
    
    for metric in higher_better:
        idx = results_df[metric].idxmax()
        best_row = results_df.loc[idx]
        best_configs[metric] = {
            'feature_type': best_row['feature_type'],
            'clustering_method': best_row['clustering_method'],
            'score': best_row[metric]
        }
        
        if verbose:
            print(f"\n{metric.upper()}:")
            print(f"  Best: {best_row['feature_type']} + {best_row['clustering_method']}")
            print(f"  Score: {best_row[metric]:.4f}")
    
    for metric in lower_better:
        idx = results_df[metric].idxmin()
        best_row = results_df.loc[idx]
        best_configs[metric] = {
            'feature_type': best_row['feature_type'],
            'clustering_method': best_row['clustering_method'],
            'score': best_row[metric]
        }
        
        if verbose:
            print(f"\n{metric.upper()}:")
            print(f"  Best: {best_row['feature_type']} + {best_row['clustering_method']}")
            print(f"  Score: {best_row[metric]:.4f}")
    
    return best_configs


def save_evaluation_results(results_df, best_configs,
                            results_path='results/metrics/all_experiments_metrics.csv',
                            best_path='results/metrics/best_configurations.txt',
                            verbose=True):
    """
    Save evaluation results.
    
    Args:
        results_df: Results DataFrame
        best_configs: Best configurations dictionary
        results_path: Path to save results CSV
        best_path: Path to save best configs
        verbose: Print info
    """
    # Save full results
    results_df.to_csv(results_path, index=False)
    
    # Save best configs
    with open(best_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BEST CONFIGURATIONS FOR EACH METRIC\n")
        f.write("="*70 + "\n\n")
        
        for metric, config in best_configs.items():
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Feature Type: {config['feature_type']}\n")
            f.write(f"  Clustering Method: {config['clustering_method']}\n")
            f.write(f"  Score: {config['score']:.4f}\n\n")
    
    if verbose:
        print(f"\n✓ Results saved to:")
        print(f"  - {results_path}")
        print(f"  - {best_path}")


if __name__ == "__main__":
    # Test evaluation
    print("Testing Evaluation Module...")
    print("="*70)
    
    # Generate dummy data
    n_samples = 1000
    n_features = 64
    n_clusters = 10
    
    features = np.random.randn(n_samples, n_features)
    clusters = np.random.randint(0, n_clusters, n_samples)
    true_labels = np.random.randint(0, n_clusters, n_samples)
    
    # Test evaluation
    metrics = evaluate_clustering_full(features, clusters, true_labels, verbose=True)
    
    print("\nEvaluation Test Complete!")
