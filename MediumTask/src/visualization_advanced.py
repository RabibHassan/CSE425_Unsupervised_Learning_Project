"""
Advanced Visualization Module
Creates comparison plots for multi-modal clustering results
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.manifold import TSNE # type: ignore
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_tsne_comparison(features_dict, clustering_results, labels,
                        save_path='results/visualizations/tsne_comparison.png',
                        verbose=True):
    """
    Create t-SNE plots comparing different feature types.
    
    Args:
        features_dict: Dictionary of feature variants
        clustering_results: Clustering results for each variant
        labels: True genre labels
        save_path: Path to save figure
        verbose: Print info
    """
    if verbose:
        print("\nCreating t-SNE comparison plots...")
    
    n_variants = len(features_dict)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (name, features) in enumerate(features_dict.items()):
        if idx >= 6:  # Max 6 plots
            break
        
        ax = axes[idx]
        
        # Compute t-SNE
        if verbose:
            print(f"  Computing t-SNE for {name}...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_features = tsne.fit_transform(features)
        
        # Get clusters for K-Means (default visualization)
        if name in clustering_results and 'kmeans' in clustering_results[name]:
            clusters = clustering_results[name]['kmeans']['clusters']
        else:
            clusters = np.zeros(len(features))
        
        # Plot
        scatter = ax.scatter(
            tsne_features[:, 0], tsne_features[:, 1],
            c=clusters, cmap='tab10', alpha=0.6, s=20,
            edgecolors='black', linewidth=0.3
        )
        
        ax.set_title(f'{name}\n(K-Means Clustering)', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Hide unused subplots
    for idx in range(len(features_dict), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✓ t-SNE comparison saved to {save_path}")


def plot_metrics_comparison(results_df,
                        save_path='results/visualizations/metrics_comparison.png',
                        verbose=True):
    """
    Create bar charts comparing metrics across methods.
    
    Args:
        results_df: Results DataFrame
        save_path: Path to save figure
        verbose: Print info
    """
    if verbose:
        print("\nCreating metrics comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metrics = [
        'silhouette_score',
        'calinski_harabasz_index',
        'davies_bouldin_index',
        'adjusted_rand_index',
        'normalized_mutual_info',
        'cluster_purity'
    ]
    
    titles = [
        'Silhouette Score ↑',
        'Calinski-Harabasz Index ↑',
        'Davies-Bouldin Index ↓',
        'Adjusted Rand Index ↑',
        'Normalized Mutual Info ↑',
        'Cluster Purity ↑'
    ]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Pivot data for grouped bar chart
        pivot_data = results_df.pivot_table(
            values=metric,
            index='feature_type',
            columns='clustering_method',
            aggfunc='mean'
        )
        
        # Plot
        pivot_data.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Type', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.legend(title='Clustering Method', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✓ Metrics comparison saved to {save_path}")


def plot_best_methods_summary(results_df,
                            save_path='results/visualizations/best_methods_summary.png',
                            verbose=True):
    """
    Create summary plot showing best methods.
    
    Args:
        results_df: Results DataFrame
        save_path: Path to save figure
        verbose: Print info
    """
    if verbose:
        print("\nCreating best methods summary...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Find best configuration for each feature type
    best_rows = []
    for feature_type in results_df['feature_type'].unique():
        subset = results_df[results_df['feature_type'] == feature_type]
        best_idx = subset['silhouette_score'].idxmax()
        best_rows.append(subset.loc[best_idx])
    
    best_df = pd.DataFrame(best_rows)
    
    # Create grouped bar chart
    x = np.arange(len(best_df))
    width = 0.15
    
    metrics_to_plot = [
        ('silhouette_score', 'Silhouette'),
        ('adjusted_rand_index', 'ARI'),
        ('normalized_mutual_info', 'NMI'),
        ('cluster_purity', 'Purity')
    ]
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']
    
    for i, ((metric, label), color) in enumerate(zip(metrics_to_plot, colors)):
        offset = width * (i - 1.5)
        bars = ax.bar(
            x + offset,
            best_df[metric],
            width,
            label=label,
            color=color,
            edgecolor='black',
            linewidth=1
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    ax.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Clustering Performance by Feature Type', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(best_df['feature_type'], rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✓ Best methods summary saved to {save_path}")


def plot_method_heatmap(results_df,
                       save_path='results/visualizations/method_heatmap.png',
                       verbose=True):
    """
    Create heatmap showing performance across methods and features.
    
    Args:
        results_df: Results DataFrame
        save_path: Path to save figure
        verbose: Print info
    """
    if verbose:
        print("\nCreating method performance heatmap...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap 1: Silhouette Score
    pivot_sil = results_df.pivot_table(
        values='silhouette_score',
        index='feature_type',
        columns='clustering_method',
        aggfunc='mean'
    )
    
    sns.heatmap(
        pivot_sil,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        ax=axes[0],
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    axes[0].set_title('Silhouette Score by Method', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Clustering Method')
    axes[0].set_ylabel('Feature Type')
    
    # Heatmap 2: Adjusted Rand Index
    pivot_ari = results_df.pivot_table(
        values='adjusted_rand_index',
        index='feature_type',
        columns='clustering_method',
        aggfunc='mean'
    )
    
    sns.heatmap(
        pivot_ari,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        ax=axes[1],
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    axes[1].set_title('Adjusted Rand Index by Method', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Clustering Method')
    axes[1].set_ylabel('Feature Type')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✓ Method heatmap saved to {save_path}")


def create_all_visualizations(features_dict, clustering_results, results_df, labels,
                             output_dir='results/visualizations',
                             verbose=True):
    """
    Create all visualization plots.
    
    Args:
        features_dict: Dictionary of feature variants
        clustering_results: Clustering results
        results_df: Evaluation results DataFrame
        labels: True genre labels
        output_dir: Output directory
        verbose: Print info
    """
    if verbose:
        print("\n" + "="*70)
        print("CREATING ALL VISUALIZATIONS")
        print("="*70)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. t-SNE comparison
    plot_tsne_comparison(
        features_dict, clustering_results, labels,
        save_path=f'{output_dir}/tsne_comparison.png',
        verbose=verbose
    )
    
    # 2. Metrics comparison
    plot_metrics_comparison(
        results_df,
        save_path=f'{output_dir}/metrics_comparison.png',
        verbose=verbose
    )
    
    # 3. Best methods summary
    plot_best_methods_summary(
        results_df,
        save_path=f'{output_dir}/best_methods_summary.png',
        verbose=verbose
    )
    
    # 4. Method heatmap
    plot_method_heatmap(
        results_df,
        save_path=f'{output_dir}/method_heatmap.png',
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "="*70)
        print("✓ All visualizations created!")
        print("="*70)


if __name__ == "__main__":
    # Test visualization
    print("Testing Visualization Module...")
    print("="*70)
    
    # Generate dummy data
    n_samples = 1000
    features_dict = {
        'audio_only': np.random.randn(n_samples, 32),
        'lyrics_only': np.random.randn(n_samples, 32),
        'concat': np.random.randn(n_samples, 64)
    }
    
    clustering_results = {}
    for name in features_dict.keys():
        clustering_results[name] = {
            'kmeans': {'clusters': np.random.randint(0, 10, n_samples)}
        }
    
    results_data = []
    for feature in features_dict.keys():
        for method in ['kmeans', 'agglomerative', 'dbscan']:
            results_data.append({
                'feature_type': feature,
                'clustering_method': method,
                'silhouette_score': np.random.rand(),
                'calinski_harabasz_index': np.random.rand() * 1000,
                'davies_bouldin_index': np.random.rand() * 2,
                'adjusted_rand_index': np.random.rand(),
                'normalized_mutual_info': np.random.rand(),
                'cluster_purity': np.random.rand()
            })
    
    results_df = pd.DataFrame(results_data)
    labels = np.random.randint(0, 10, n_samples)
    
    # Create visualizations
    create_all_visualizations(
        features_dict, clustering_results, results_df, labels,
        output_dir='results/visualizations',
        verbose=True
    )
    
    print("\nVisualization Test Complete!")
