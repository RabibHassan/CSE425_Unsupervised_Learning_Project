"""
Multi-Modal Fusion Module
Combines audio and lyrics latent representations
"""

import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import joblib # type: ignore


def concatenate_features(audio_latent, lyrics_latent, 
                        normalize=True, verbose=True):
    """
    Simple concatenation of audio and lyrics features.
    
    Args:
        audio_latent: Audio latent features (n_samples, latent_dim_audio)
        lyrics_latent: Lyrics latent features (n_samples, latent_dim_lyrics)
        normalize: Normalize combined features
        verbose: Print information
        
    Returns:
        combined_features: Concatenated features
        scaler: Fitted scaler (if normalize=True)
    """
    if verbose:
        print("\n" + "="*70)
        print("MULTI-MODAL FEATURE FUSION")
        print("="*70)
        print(f"Audio latent shape: {audio_latent.shape}")
        print(f"Lyrics latent shape: {lyrics_latent.shape}")
    
    # Ensure same number of samples
    assert audio_latent.shape[0] == lyrics_latent.shape[0], \
        "Audio and lyrics must have same number of samples"
    
    # Concatenate
    combined_features = np.concatenate([audio_latent, lyrics_latent], axis=1)
    
    if verbose:
        print(f"\n✓ Combined features shape: {combined_features.shape}")
        print(f"  Total dimensions: {combined_features.shape[1]}")
    
    # Normalize if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
        
        if verbose:
            print(f"\n✓ Normalized combined features:")
            print(f"  Mean: {combined_features.mean():.6f}")
            print(f"  Std: {combined_features.std():.6f}")
    
    return combined_features, scaler


def weighted_fusion(audio_latent, lyrics_latent,
                   audio_weight=0.5, lyrics_weight=0.5,
                   normalize=True, verbose=True):
    """
    Weighted combination of audio and lyrics features.
    
    Args:
        audio_latent: Audio latent features
        lyrics_latent: Lyrics latent features
        audio_weight: Weight for audio features
        lyrics_weight: Weight for lyrics features
        normalize: Normalize after weighting
        verbose: Print information
        
    Returns:
        combined_features: Weighted combination
        scaler: Fitted scaler (if normalize=True)
    """
    if verbose:
        print("\n" + "="*70)
        print("WEIGHTED MULTI-MODAL FUSION")
        print("="*70)
        print(f"Audio weight: {audio_weight}")
        print(f"Lyrics weight: {lyrics_weight}")
    
    # Normalize weights
    total_weight = audio_weight + lyrics_weight
    audio_weight = audio_weight / total_weight
    lyrics_weight = lyrics_weight / total_weight
    
    # Weighted concatenation
    audio_weighted = audio_latent * audio_weight
    lyrics_weighted = lyrics_latent * lyrics_weight
    
    combined_features = np.concatenate([audio_weighted, lyrics_weighted], axis=1)
    
    if verbose:
        print(f"\n✓ Weighted features shape: {combined_features.shape}")
    
    # Normalize
    scaler = None
    if normalize:
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
    
    return combined_features, scaler


def average_fusion(audio_latent, lyrics_latent, verbose=True):
    """
    Average audio and lyrics features (element-wise mean).
    Requires same dimensions.
    
    Args:
        audio_latent: Audio latent features
        lyrics_latent: Lyrics latent features
        verbose: Print information
        
    Returns:
        combined_features: Averaged features
    """
    if verbose:
        print("\n" + "="*70)
        print("AVERAGE MULTI-MODAL FUSION")
        print("="*70)
    
    assert audio_latent.shape == lyrics_latent.shape, \
        "Audio and lyrics must have same shape for averaging"
    
    combined_features = (audio_latent + lyrics_latent) / 2.0
    
    if verbose:
        print(f"✓ Averaged features shape: {combined_features.shape}")
    
    return combined_features


def create_multimodal_variants(audio_latent, lyrics_latent, verbose=True):
    """
    Create multiple fusion variants for comparison.
    
    Args:
        audio_latent: Audio latent features
        lyrics_latent: Lyrics latent features
        verbose: Print information
        
    Returns:
        fusion_dict: Dictionary with different fusion methods
    """
    if verbose:
        print("\n" + "="*70)
        print("CREATING MULTI-MODAL VARIANTS")
        print("="*70)
    
    fusion_dict = {}
    
    # 1. Audio only
    fusion_dict['audio_only'] = audio_latent
    if verbose:
        print(f"✓ Audio only: {audio_latent.shape}")
    
    # 2. Lyrics only
    fusion_dict['lyrics_only'] = lyrics_latent
    if verbose:
        print(f"✓ Lyrics only: {lyrics_latent.shape}")
    
    # 3. Simple concatenation
    concat_features, _ = concatenate_features(
        audio_latent, lyrics_latent, 
        normalize=True, verbose=False
    )
    fusion_dict['concat'] = concat_features
    if verbose:
        print(f"✓ Concatenation: {concat_features.shape}")
    
    # 4. Weighted (audio-heavy)
    audio_heavy, _ = weighted_fusion(
        audio_latent, lyrics_latent,
        audio_weight=0.7, lyrics_weight=0.3,
        normalize=True, verbose=False
    )
    fusion_dict['audio_heavy'] = audio_heavy
    if verbose:
        print(f"✓ Audio-heavy (70/30): {audio_heavy.shape}")
    
    # 5. Weighted (lyrics-heavy)
    lyrics_heavy, _ = weighted_fusion(
        audio_latent, lyrics_latent,
        audio_weight=0.3, lyrics_weight=0.7,
        normalize=True, verbose=False
    )
    fusion_dict['lyrics_heavy'] = lyrics_heavy
    if verbose:
        print(f"✓ Lyrics-heavy (30/70): {lyrics_heavy.shape}")
    
    # 6. Equal weighted
    equal_weighted, _ = weighted_fusion(
        audio_latent, lyrics_latent,
        audio_weight=0.5, lyrics_weight=0.5,
        normalize=True, verbose=False
    )
    fusion_dict['equal_weighted'] = equal_weighted
    if verbose:
        print(f"✓ Equal weighted (50/50): {equal_weighted.shape}")
    
    if verbose:
        print("\n" + "="*70)
        print(f"✓ Created {len(fusion_dict)} fusion variants")
        print("="*70)
    
    return fusion_dict


def save_fusion_results(fusion_dict, labels, 
                       output_path='data/processed/multimodal_features.pkl',
                       verbose=True):
    """
    Save all fusion variants.
    
    Args:
        fusion_dict: Dictionary of fusion variants
        labels: Genre labels
        output_path: Save path
        verbose: Print information
    """
    fusion_data = {
        'features': fusion_dict,
        'labels': labels,
        'n_variants': len(fusion_dict)
    }
    
    joblib.dump(fusion_data, output_path)
    
    if verbose:
        print(f"\n✓ Fusion results saved to {output_path}")


if __name__ == "__main__":
    # Test fusion
    print("Testing Multi-Modal Fusion...")
    print("="*70)
    
    # Create dummy data
    n_samples = 1000
    audio_latent = np.random.randn(n_samples, 32)
    lyrics_latent = np.random.randn(n_samples, 32)
    
    # Test fusion methods
    fusion_dict = create_multimodal_variants(
        audio_latent, lyrics_latent, verbose=True
    )
    
    print("\nFusion Test Complete!")
