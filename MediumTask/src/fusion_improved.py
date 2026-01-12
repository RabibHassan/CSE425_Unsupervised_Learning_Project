"""
fusion_improved.py

Advanced Multi-Modal Fusion Strategies
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def concat_with_pca(audio_latent, lyrics_latent, target_dim=24, verbose=True):
    """
    Concatenate audio and lyrics latent features, then apply PCA.
    
    Args:
        audio_latent: Audio VAE latent features (n_samples, audio_dim)
        lyrics_latent: Lyrics VAE latent features (n_samples, lyrics_dim)
        target_dim: Target dimensionality after PCA
        verbose: Print progress
    
    Returns:
        fused_features: Fused and reduced features (n_samples, target_dim)
        pca_model: Fitted PCA model
    """
    if verbose:
        print("\n" + "="*70)
        print("IMPROVED FUSION: CONCATENATION + PCA")
        print("="*70)
    
    # Concatenate features
    concat_features = np.hstack([audio_latent, lyrics_latent])
    if verbose:
        print(f"Concatenated shape: {concat_features.shape}")
    
    # Apply PCA
    pca = PCA(n_components=target_dim, random_state=42)
    fused_features = pca.fit_transform(concat_features)
    
    if verbose:
        print(f"\n✓ PCA applied:")
        print(f"  Original dimensions: {concat_features.shape[1]}")
        print(f"  Reduced dimensions: {target_dim}")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"  Final shape: {fused_features.shape}")
    
    return fused_features, pca


def weighted_concat_with_pca(audio_latent, lyrics_latent, 
                              audio_weight=0.7, lyrics_weight=0.3,
                              target_dim=16, verbose=True):
    """
    Weighted concatenation of audio and lyrics, then apply PCA.
    
    Args:
        audio_latent: Audio VAE latent features
        lyrics_latent: Lyrics VAE latent features
        audio_weight: Weight for audio features (0-1)
        lyrics_weight: Weight for lyrics features (0-1)
        target_dim: Target dimensionality after PCA
        verbose: Print progress
    
    Returns:
        fused_features: Weighted fused features (n_samples, target_dim)
        pca_model: Fitted PCA model
    """
    if verbose:
        print("\n" + "="*70)
        print(f"WEIGHTED FUSION: Audio {audio_weight} | Lyrics {lyrics_weight}")
        print("="*70)
    
    # Standardize features first
    audio_scaled = StandardScaler().fit_transform(audio_latent)
    lyrics_scaled = StandardScaler().fit_transform(lyrics_latent)
    
    # Apply weights
    audio_weighted = audio_scaled * audio_weight
    lyrics_weighted = lyrics_scaled * lyrics_weight
    
    # Concatenate weighted features
    concat_weighted = np.hstack([audio_weighted, lyrics_weighted])
    
    # Apply PCA
    pca = PCA(n_components=target_dim, random_state=42)
    fused_features = pca.fit_transform(concat_weighted)
    
    if verbose:
        print(f"✓ Fused shape: {fused_features.shape}")
    
    return fused_features, pca


def simple_concat(audio_latent, lyrics_latent, verbose=True):
    """
    Simple concatenation without dimensionality reduction.
    
    Args:
        audio_latent: Audio VAE latent features
        lyrics_latent: Lyrics VAE latent features
        verbose: Print progress
    
    Returns:
        concatenated features
    """
    concat = np.hstack([audio_latent, lyrics_latent])
    if verbose:
        print(f"✓ Simple concatenation: {concat.shape}")
    return concat
