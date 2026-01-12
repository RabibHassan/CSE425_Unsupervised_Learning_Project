"""
Reconstruction Visualization Module
Generates before/after reconstruction comparisons
Author: Moin Mostakim
Hard Task - Neural Networks Project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_reconstruction_comparison(original, reconstructed, feature_names=None,
                                   n_samples=5, output_path='reconstruction.png',
                                   title='Feature Reconstruction'):
    """
    Plot side-by-side comparison of original vs reconstructed features.
    
    Args:
        original: Original features (n_samples, n_features)
        reconstructed: Reconstructed features (n_samples, n_features)
        feature_names: Names of features (optional)
        n_samples: Number of samples to visualize
        output_path: Save path
        title: Plot title
    """
    n_samples = min(n_samples, original.shape[0])
    n_features = original.shape[1]
    
    # Select random samples
    indices = np.random.choice(original.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Original
        axes[i, 0].bar(range(n_features), original[idx], color='steelblue', alpha=0.7)
        axes[i, 0].set_title(f'Sample {idx} - Original', fontsize=10, weight='bold')
        axes[i, 0].set_xlabel('Feature Index')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(axis='y', alpha=0.3)
        
        # Reconstructed
        axes[i, 1].bar(range(n_features), reconstructed[idx], color='coral', alpha=0.7)
        axes[i, 1].set_title(f'Sample {idx} - Reconstructed', fontsize=10, weight='bold')
        axes[i, 1].set_xlabel('Feature Index')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(axis='y', alpha=0.3)
        
        # Compute MSE for this sample
        mse = np.mean((original[idx] - reconstructed[idx])**2)
        axes[i, 1].text(0.95, 0.95, f'MSE: {mse:.4f}',
                       transform=axes[i, 1].transAxes,
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Reconstruction plot saved to {output_path}")


def plot_reconstruction_heatmap(original, reconstructed, n_samples=10,
                                output_path='reconstruction_heatmap.png',
                                title='Reconstruction Heatmap'):
    """
    Plot heatmap showing reconstruction quality.
    
    Args:
        original: Original features
        reconstructed: Reconstructed features
        n_samples: Number of samples
        output_path: Save path
        title: Plot title
    """
    n_samples = min(n_samples, original.shape[0])
    indices = np.random.choice(original.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    sns.heatmap(original[indices], ax=axes[0], cmap='viridis',
                cbar_kws={'label': 'Value'})
    axes[0].set_title('Original Features', fontsize=12, weight='bold')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Sample Index')
    
    # Reconstructed
    sns.heatmap(reconstructed[indices], ax=axes[1], cmap='viridis',
                cbar_kws={'label': 'Value'})
    axes[1].set_title('Reconstructed Features', fontsize=12, weight='bold')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Sample Index')
    
    # Error
    error = np.abs(original[indices] - reconstructed[indices])
    sns.heatmap(error, ax=axes[2], cmap='Reds',
                cbar_kws={'label': 'Absolute Error'})
    axes[2].set_title('Reconstruction Error', fontsize=12, weight='bold')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Sample Index')
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Reconstruction heatmap saved to {output_path}")


def plot_model_reconstruction_comparison(models_dict, features, labels,
                                        output_path='model_comparison.png'):
    """
    Compare reconstruction quality across multiple models.
    
    Args:
        models_dict: Dict of {model_name: model_object}
        features: Original features
        labels: Labels (for CVAE)
        output_path: Save path
    """
    n_models = len(models_dict)
    
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4*(n_models+1), 5))
    
    # Original
    sample_idx = np.random.randint(0, len(features))
    
    axes[0].bar(range(features.shape[1]), features[sample_idx], 
                color='steelblue', alpha=0.7)
    axes[0].set_title('Original', fontsize=12, weight='bold')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Value')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Reconstructions
    for i, (model_name, model) in enumerate(models_dict.items()):
        # Handle CVAE (needs labels)
        if hasattr(model, 'reconstruct') and 'cvae' in model_name.lower():
            recon = model.reconstruct(features[sample_idx:sample_idx+1], 
                                     labels[sample_idx:sample_idx+1])
        else:
            recon = model.reconstruct(features[sample_idx:sample_idx+1])
        
        mse = np.mean((features[sample_idx] - recon[0])**2)
        
        axes[i+1].bar(range(features.shape[1]), recon[0], 
                     color='coral', alpha=0.7)
        axes[i+1].set_title(f'{model_name}\nMSE: {mse:.4f}', 
                           fontsize=10, weight='bold')
        axes[i+1].set_xlabel('Feature Index')
        axes[i+1].set_ylabel('Value')
        axes[i+1].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Reconstruction Comparison (Sample {sample_idx})',
                fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model comparison plot saved to {output_path}")


def save_reconstruction_examples(model, features, labels=None, n_samples=10,
                                 output_dir='results/reconstructions'):
    """
    Save reconstruction examples as numpy arrays.
    
    Args:
        model: Trained model
        features: Original features
        labels: Labels (for CVAE)
        n_samples: Number of samples
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    indices = np.random.choice(len(features), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        original = features[idx:idx+1]
        
        # Handle CVAE
        if labels is not None and hasattr(model, 'reconstruct'):
            reconstructed = model.reconstruct(original, labels[idx:idx+1])
        else:
            reconstructed = model.reconstruct(original)
        
        # Save
        np.save(f'{output_dir}/sample_{i}_original.npy', original)
        np.save(f'{output_dir}/sample_{i}_reconstructed.npy', reconstructed)
    
    print(f"✓ {n_samples} reconstruction examples saved to {output_dir}")
