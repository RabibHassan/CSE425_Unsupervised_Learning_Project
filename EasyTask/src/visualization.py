"""
Visualization Utilities
Functions for creating t-SNE plots and result visualizations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_tsne_clusters(features, clusters, title="Cluster Visualization", 
                       save_path=None):
    """
    Create t-SNE visualization of clusters.

    Args:
        features: Feature matrix
        clusters: Cluster assignments
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Apply t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_features = tsne.fit_transform(features)

    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6, s=30, 
                         edgecolors='black', linewidth=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")

    plt.close()  # Close instead of show

    return tsne_features


def plot_comparison_visualization(vae_features, pca_features, 
                                  vae_clusters, pca_clusters,
                                  language_labels, genre_labels,
                                  save_path=None):
    """
    Create comprehensive comparison visualization.

    Args:
        vae_features: VAE latent features
        pca_features: PCA features
        vae_clusters: VAE cluster assignments
        pca_clusters: PCA cluster assignments
        language_labels: True language labels
        genre_labels: True genre labels
        save_path: Path to save figure
    """
    print("Creating comprehensive visualization...")

    # Compute t-SNE for both
    print("Computing t-SNE for VAE features...")
    tsne_vae = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_features_vae = tsne_vae.fit_transform(vae_features)

    print("Computing t-SNE for PCA features...")
    tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_features_pca = tsne_pca.fit_transform(pca_features)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Convert language labels to numeric
    language_numeric = np.array([0 if lang == 'Bangla' else 1 for lang in language_labels])

    # Convert genre labels to numeric
    genre_unique = np.unique(genre_labels)
    genre_numeric = np.array([np.where(genre_unique == g)[0][0] for g in genre_labels])

    # Plot 1: VAE clusters
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(tsne_features_vae[:, 0], tsne_features_vae[:, 1], 
                          c=vae_clusters, cmap='viridis', alpha=0.6, s=30, 
                          edgecolors='black', linewidth=0.5)
    ax1.set_title('VAE + K-Means Clustering\n(t-SNE Visualization)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    ax1.grid(True, alpha=0.3)

    # Plot 2: VAE colored by language
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(tsne_features_vae[:, 0], tsne_features_vae[:, 1], 
                          c=language_numeric, cmap='coolwarm', alpha=0.6, s=30, 
                          edgecolors='black', linewidth=0.5)
    ax2.set_title('True Language Labels\n(VAE Latent Space)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(['Bangla', 'English'])
    ax2.grid(True, alpha=0.3)

    # Plot 3: VAE colored by genre
    ax3 = plt.subplot(2, 3, 3)
    scatter3 = ax3.scatter(tsne_features_vae[:, 0], tsne_features_vae[:, 1], 
                          c=genre_numeric, cmap='tab20', alpha=0.6, s=30, 
                          edgecolors='black', linewidth=0.5)
    ax3.set_title('Genre Distribution\n(VAE Latent Space)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('t-SNE Component 1')
    ax3.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter3, ax=ax3, label='Genre')
    ax3.grid(True, alpha=0.3)

    # Plot 4: PCA clusters
    ax4 = plt.subplot(2, 3, 4)
    scatter4 = ax4.scatter(tsne_features_pca[:, 0], tsne_features_pca[:, 1], 
                          c=pca_clusters, cmap='viridis', alpha=0.6, s=30, 
                          edgecolors='black', linewidth=0.5)
    ax4.set_title('PCA + K-Means Clustering\n(t-SNE Visualization)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('t-SNE Component 1')
    ax4.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter4, ax=ax4, label='Cluster')
    ax4.grid(True, alpha=0.3)

    # Plot 5: PCA colored by language
    ax5 = plt.subplot(2, 3, 5)
    scatter5 = ax5.scatter(tsne_features_pca[:, 0], tsne_features_pca[:, 1], 
                          c=language_numeric, cmap='coolwarm', alpha=0.6, s=30, 
                          edgecolors='black', linewidth=0.5)
    ax5.set_title('True Language Labels\n(PCA Space)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('t-SNE Component 1')
    ax5.set_ylabel('t-SNE Component 2')
    cbar5 = plt.colorbar(scatter5, ax=ax5)
    cbar5.set_ticks([0, 1])
    cbar5.set_ticklabels(['Bangla', 'English'])
    ax5.grid(True, alpha=0.3)

    # Plot 6: Metrics comparison (placeholder - will be filled by main script)
    ax6 = plt.subplot(2, 3, 6)
    ax6.text(0.5, 0.5, 'Metrics Comparison\n(See metrics_comparison.png)', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax6.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive visualization saved to {save_path}")

    plt.close()  # Close instead of show

    return tsne_features_vae, tsne_features_pca


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot bar chart comparing different methods.

    Args:
        metrics_dict: Dictionary mapping method names to their metrics
        save_path: Path to save figure
    """
    methods = list(metrics_dict.keys())

    # Extract metrics
    sil_scores = [metrics_dict[m]['silhouette_score'] for m in methods]
    ch_scores = [metrics_dict[m]['calinski_harabasz_index'] / 1000 for m in methods]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, sil_scores, width, label='Silhouette Score', 
                   color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, ch_scores, width, label='CH Index (÷1000)', 
                   color='coral', edgecolor='black')

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {save_path}")

    plt.close()  # Close instead of show


def plot_latent_space_distribution(latent_features, labels, label_type="Language",
                                   save_path=None):
    """
    Plot distribution of samples in latent space.

    Args:
        latent_features: Latent representations
        labels: Sample labels
        label_type: Type of labels (e.g., "Language", "Genre")
        save_path: Path to save figure
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_features = tsne.fit_transform(latent_features)

    # Create plot
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.6, s=30, 
                   edgecolors='black', linewidth=0.5)

    plt.title(f'{label_type} Distribution in Latent Space', 
             fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ {label_type} distribution plot saved to {save_path}")

    plt.close()  # Close instead of show