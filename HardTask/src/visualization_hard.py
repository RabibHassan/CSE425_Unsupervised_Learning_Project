"""
Advanced Visualization Module for Hard Task
Genre distribution, latent space analysis, comprehensive comparisons
Author: Moin Mostakim
Hard Task - Neural Networks Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def plot_genre_distribution(clustering_results, true_labels, genre_names,
                            output_path='genre_distribution.png'):
    """
    Plot cluster distribution across genres.
    
    Args:
        clustering_results: Dict of cluster assignments
        true_labels: True genre labels
        genre_names: List of genre names
        output_path: Save path
    """
    n_methods = len(clustering_results)
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (method_name, clusters) in enumerate(clustering_results.items()):
        # Create confusion-like matrix
        n_genres = len(np.unique(true_labels))
        n_clusters = len(np.unique(clusters))
        
        matrix = np.zeros((n_genres, n_clusters))
        
        for genre_idx in range(n_genres):
            genre_mask = true_labels == genre_idx
            genre_clusters = clusters[genre_mask]
            
            for cluster_idx in range(n_clusters):
                matrix[genre_idx, cluster_idx] = np.sum(genre_clusters == cluster_idx)
        
        # Normalize by row (genre)
        matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)
        
        # Plot
        sns.heatmap(matrix_norm, ax=axes[idx], cmap='YlOrRd',
                   xticklabels=range(n_clusters),
                   yticklabels=genre_names,
                   cbar_kws={'label': 'Proportion'})
        
        axes[idx].set_title(f'{method_name}', fontsize=11, weight='bold')
        axes[idx].set_xlabel('Cluster ID')
        axes[idx].set_ylabel('Genre')
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Genre Distribution Across Clusters', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Genre distribution plot saved to {output_path}")


def plot_latent_space_cvae(cvae_model, features, labels, genre_names,
                           output_path='latent_space_cvae.png'):
    """
    Visualize CVAE latent space colored by genre.
    
    Args:
        cvae_model: Trained CVAE model
        features: Input features
        labels: Genre labels
        genre_names: List of genre names
        output_path: Save path
    """
    # Encode features
    latent = cvae_model.encode(features, labels)
    
    # Apply t-SNE if latent_dim > 2
    if latent.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent)
    else:
        latent_2d = latent
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                        c=labels, cmap='tab10', alpha=0.6, s=50)
    
    # Add legend
    handles, _ = scatter.legend_elements()
    ax.legend(handles, genre_names, title='Genre',
             loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.set_title('CVAE Latent Space (t-SNE projection)', 
                fontsize=14, weight='bold')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CVAE latent space plot saved to {output_path}")


def plot_comprehensive_comparison(results_df, output_path='comprehensive_comparison.png'):
    """
    Create comprehensive comparison across all methods.
    
    Args:
        results_df: DataFrame with all results
        output_path: Save path
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics = ['silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index',
               'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity']
    
    titles = ['Silhouette Score ↑', 'Calinski-Harabasz Index ↑', 'Davies-Bouldin Index ↓',
              'Adjusted Rand Index ↑', 'Normalized Mutual Info ↑', 'Cluster Purity ↑']
    
    # Filter valid results (exclude DBSCAN failures)
    valid_df = results_df[results_df['n_clusters'] > 1].copy()
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Group by feature type
        pivot_data = valid_df.pivot_table(
            values=metric,
            index='feature_type',
            columns='clustering_method',
            aggfunc='mean'
        )
        
        # Plot
        pivot_data.plot(kind='bar', ax=ax, rot=45, width=0.8)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('Feature Type')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(title='Clustering', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Comprehensive Method Comparison - All Metrics',
                fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive comparison plot saved to {output_path}")


def plot_tsne_all_methods(feature_dict, labels, genre_names,
                          output_path='tsne_all_methods.png'):
    """
    t-SNE comparison for all feature extraction methods.
    
    Args:
        feature_dict: Dict of {method_name: features}
        labels: True labels
        genre_names: Genre names
        output_path: Save path
    """
    n_methods = len(feature_dict)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten()
    
    for idx, (method_name, features) in enumerate(feature_dict.items()):
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        scatter = axes[idx].scatter(features_2d[:, 0], features_2d[:, 1],
                                   c=labels, cmap='tab10', alpha=0.6, s=30)
        
        axes[idx].set_title(f'{method_name}', fontsize=12, weight='bold')
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    # Add shared legend
    handles, _ = scatter.legend_elements()
    fig.legend(handles, genre_names, title='Genre',
              loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.suptitle('t-SNE Visualization - All Feature Extraction Methods',
                fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE comparison plot saved to {output_path}")


def plot_method_comparison_radar(results_df, top_n=5,
                                output_path='method_radar.png'):
    """
    Radar chart comparing top methods across metrics.
    
    Args:
        results_df: DataFrame with results
        top_n: Number of top methods to show
        output_path: Save path
    """
    from math import pi
    
    # Select valid results
    valid_df = results_df[results_df['n_clusters'] > 1].copy()
    
    # Get top methods by ARI
    top_methods = valid_df.nlargest(top_n, 'adjusted_rand_index')
    
    # Metrics to compare (normalized 0-1)
    metrics = ['silhouette_score', 'adjusted_rand_index', 
               'normalized_mutual_info', 'cluster_purity']
    
    # Normalize metrics
    for metric in metrics:
        max_val = valid_df[metric].max()
        min_val = valid_df[metric].min()
        if max_val > min_val:
            top_methods[f'{metric}_norm'] = (top_methods[metric] - min_val) / (max_val - min_val)
        else:
            top_methods[f'{metric}_norm'] = 0.5
    
    # Setup radar chart
    categories = ['Silhouette', 'ARI', 'NMI', 'Purity']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, top_n))
    
    for idx, (_, row) in enumerate(top_methods.iterrows()):
        values = [row[f'{m}_norm'] for m in metrics]
        values += values[:1]
        
        label = f"{row['feature_type']} + {row['clustering_method']}"
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(f'Top {top_n} Methods - Normalized Performance', 
             size=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Radar chart saved to {output_path}")


def create_all_hard_visualizations(feature_dict, clustering_results, results_df,
                                   labels, genre_names, cvae_model=None,
                                   original_features=None,
                                   output_dir='results/visualizations', verbose=True):
    """
    Create all visualizations for hard task.
    
    Args:
        feature_dict: Dict of latent features
        clustering_results: Dict of clustering results
        results_df: Results DataFrame
        labels: True labels
        genre_names: Genre names
        cvae_model: Trained CVAE model (optional)
        original_features: Original audio features for CVAE visualization
        output_dir: Output directory
        verbose: Print progress
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("\n" + "="*70)
        print("CREATING ADVANCED VISUALIZATIONS")
        print("="*70)
    
    # 1. Genre distribution
    if verbose:
        print("\n1. Creating genre distribution plot...")
    
    kmeans_results = {}
    for name, results in clustering_results.items():
        if 'kmeans' in results:
            kmeans_results[name] = results['kmeans']['clusters']
    
    plot_genre_distribution(
        kmeans_results,
        labels,
        genre_names,
        output_path=f'{output_dir}/genre_distribution.png'
    )
    
    # 2. t-SNE all methods
    if verbose:
        print("\n2. Creating t-SNE comparison...")
    
    plot_tsne_all_methods(
        feature_dict,
        labels,
        genre_names,
        output_path=f'{output_dir}/tsne_all_methods.png'
    )
    
    # 3. CVAE latent space
    if cvae_model is not None and original_features is not None and verbose:
        print("\n3. Creating CVAE latent space plot...")
        plot_latent_space_cvae(
            cvae_model,
            original_features,
            labels,
            genre_names,
            output_path=f'{output_dir}/latent_space_cvae.png'
        )
    
    # 4. Comprehensive comparison
    if verbose:
        print("\n4. Creating comprehensive comparison...")
    
    plot_comprehensive_comparison(
        results_df,
        output_path=f'{output_dir}/comprehensive_comparison.png'
    )
    
    # 5. Radar chart
    if verbose:
        print("\n5. Creating radar chart...")
    
    plot_method_comparison_radar(
        results_df,
        top_n=5,
        output_path=f'{output_dir}/method_radar.png'
    )
    
    if verbose:
        print("\n" + "="*70)
        print("All visualizations created.")
        print("="*70)
