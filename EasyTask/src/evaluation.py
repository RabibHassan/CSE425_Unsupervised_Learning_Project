"""
Evaluation Metrics
Functions to compute clustering quality metrics
"""

import numpy as np # pyright: ignore[reportMissingImports]
from sklearn.metrics import silhouette_score, calinski_harabasz_score # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score # type: ignore


def evaluate_clustering(X, clusters, labels=None):
    """
    Evaluate clustering quality with multiple metrics.

    Args:
        X: Feature matrix
        clusters: Cluster assignments
        labels: True labels (optional, for supervised metrics)

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}

    # Silhouette Score (higher is better, range [-1, 1])
    sil_score = silhouette_score(X, clusters)
    metrics['silhouette_score'] = sil_score

    # Calinski-Harabasz Index (higher is better)
    ch_score = calinski_harabasz_score(X, clusters)
    metrics['calinski_harabasz_index'] = ch_score

    # Davies-Bouldin Index (lower is better)
    db_score = davies_bouldin_score(X, clusters)
    metrics['davies_bouldin_index'] = db_score

    # Adjusted Rand Index (if true labels provided)
    if labels is not None:
        ari_score = adjusted_rand_score(labels, clusters)
        metrics['adjusted_rand_index'] = ari_score

    return metrics


def print_metrics(metrics, method_name="Clustering"):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metrics
        method_name: Name of clustering method
    """
    print(f"\n{method_name} Results:")
    print("-" * 50)

    if 'silhouette_score' in metrics:
        print(f"  Silhouette Score:        {metrics['silhouette_score']:.4f}")

    if 'calinski_harabasz_index' in metrics:
        print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")

    if 'davies_bouldin_index' in metrics:
        print(f"  Davies-Bouldin Index:    {metrics['davies_bouldin_index']:.4f}")

    if 'adjusted_rand_index' in metrics:
        print(f"  Adjusted Rand Index:     {metrics['adjusted_rand_index']:.4f}")


def compare_methods(metrics_dict):
    """
    Compare multiple clustering methods.

    Args:
        metrics_dict: Dictionary mapping method names to their metrics

    Returns:
        Comparison DataFrame
    """
    import pandas as pd # type: ignore

    data = []
    for method, metrics in metrics_dict.items():
        row = {'Method': method}
        row.update(metrics)
        data.append(row)

    df = pd.DataFrame(data)
    return df


def compute_improvement(baseline_metrics, method_metrics):
    """
    Compute percentage improvement over baseline.

    Args:
        baseline_metrics: Metrics from baseline method
        method_metrics: Metrics from evaluation method

    Returns:
        Dictionary with improvement percentages
    """
    improvements = {}

    for metric in baseline_metrics:
        if metric in method_metrics:
            baseline_val = baseline_metrics[metric]
            method_val = method_metrics[metric]

            # For Davies-Bouldin, lower is better
            if 'davies_bouldin' in metric:
                improvement = ((baseline_val - method_val) / baseline_val) * 100
            else:
                improvement = ((method_val - baseline_val) / abs(baseline_val)) * 100

            improvements[f'{metric}_improvement_%'] = improvement

    return improvements