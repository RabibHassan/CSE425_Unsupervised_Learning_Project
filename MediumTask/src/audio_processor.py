"""
Audio Feature Processor
Extracts and preprocesses audio features from GTZAN dataset
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import joblib # type: ignore


def extract_audio_features(gtzan_df, feature_cols, verbose=True):
    """
    Extract audio features from GTZAN DataFrame.
    
    Args:
        gtzan_df: GTZAN DataFrame
        feature_cols: List of feature column names
        verbose: Print information
        
    Returns:
        features: NumPy array of audio features
        feature_names: List of feature names
    """
    if verbose:
        print("\nExtracting audio features...")
    
    # Extract feature matrix
    features = gtzan_df[feature_cols].values
    
    if verbose:
        print(f"✓ Audio feature matrix shape: {features.shape}")
        print(f"  Features per song: {features.shape[1]}")
        print(f"  Total songs: {features.shape[0]}")
        print(f"  Feature statistics:")
        print(f"    Mean: {features.mean():.4f}")
        print(f"    Std: {features.std():.4f}")
        print(f"    Min: {features.min():.4f}")
        print(f"    Max: {features.max():.4f}")
    
    return features, feature_cols


def preprocess_audio_features(features, scaler=None, save_scaler=True, 
                              scaler_path='models/audio_scaler.pkl',
                              verbose=True):
    """
    Standardize audio features (zero mean, unit variance).
    
    Args:
        features: Raw audio features
        scaler: Existing scaler (if None, creates new one)
        save_scaler: Whether to save the scaler
        scaler_path: Path to save scaler
        verbose: Print information
        
    Returns:
        features_scaled: Standardized features
        scaler: Fitted StandardScaler
    """
    if verbose:
        print("\nPreprocessing audio features...")
    
    if scaler is None:
        # Create and fit new scaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if save_scaler:
            joblib.dump(scaler, scaler_path)
            if verbose:
                print(f"✓ Scaler saved to {scaler_path}")
    else:
        # Use existing scaler
        features_scaled = scaler.transform(features)
    
    if verbose:
        print(f"✓ Standardized features:")
        print(f"  Mean: {features_scaled.mean():.6f}")
        print(f"  Std: {features_scaled.std():.6f}")
        print(f"  Shape: {features_scaled.shape}")
    
    return features_scaled, scaler


def get_audio_statistics(features, feature_names, top_n=10, verbose=True):
    """
    Get statistics and most important features.
    
    Args:
        features: Feature matrix
        feature_names: List of feature names
        top_n: Number of top features to show
        verbose: Print information
        
    Returns:
        stats_df: DataFrame with feature statistics
    """
    if verbose:
        print("\nComputing audio feature statistics...")
    
    # Calculate statistics for each feature
    stats = {
        'feature': feature_names,
        'mean': features.mean(axis=0),
        'std': features.std(axis=0),
        'min': features.min(axis=0),
        'max': features.max(axis=0),
        'variance': features.var(axis=0)
    }
    
    stats_df = pd.DataFrame(stats)
    
    # Sort by variance (most informative features)
    stats_df = stats_df.sort_values('variance', ascending=False)
    
    if verbose:
        print(f"\n✓ Top {top_n} most variant features:")
        print(stats_df.head(top_n)[['feature', 'variance', 'mean', 'std']].to_string(index=False))
    
    return stats_df


def reshape_for_conv1d(features, verbose=True):
    """
    Reshape audio features for Conv1D VAE.
    From (n_samples, n_features) to (n_samples, n_features, 1)
    
    Args:
        features: 2D feature matrix
        verbose: Print information
        
    Returns:
        features_reshaped: 3D array for Conv1D
    """
    if verbose:
        print("\nReshaping features for Convolutional VAE...")
    
    # Add channel dimension for Conv1D
    features_reshaped = features.reshape(features.shape[0], features.shape[1], 1)
    
    if verbose:
        print(f"✓ Original shape: {features.shape}")
        print(f"✓ Reshaped for Conv1D: {features_reshaped.shape}")
        print(f"  (samples, features, channels)")
    
    return features_reshaped


def process_audio_pipeline(gtzan_df, feature_cols, 
                           reshape_conv=False,
                           save_processed=True,
                           output_path='data/processed/audio_features.pkl',
                           verbose=True):
    """
    Complete audio processing pipeline.
    
    Args:
        gtzan_df: GTZAN DataFrame
        feature_cols: List of feature columns
        reshape_conv: Reshape for Conv1D
        save_processed: Save processed features
        output_path: Path to save processed data
        verbose: Print information
        
    Returns:
        features_processed: Processed audio features
        scaler: Fitted scaler
        stats: Feature statistics
    """
    if verbose:
        print("\n" + "="*70)
        print("AUDIO PROCESSING PIPELINE")
        print("="*70)
    
    # Step 1: Extract features
    features_raw, feature_names = extract_audio_features(
        gtzan_df, feature_cols, verbose=verbose
    )
    
    # Step 2: Standardize
    features_scaled, scaler = preprocess_audio_features(
        features_raw, verbose=verbose
    )
    
    # Step 3: Get statistics
    stats = get_audio_statistics(
        features_scaled, feature_names, verbose=verbose
    )
    
    # Step 4: Reshape if needed
    if reshape_conv:
        features_processed = reshape_for_conv1d(features_scaled, verbose=verbose)
    else:
        features_processed = features_scaled
    
    # Step 5: Save
    if save_processed:
        processed_data = {
            'features': features_processed,
            'feature_names': feature_names,
            'scaler': scaler,
            'stats': stats,
            'shape': features_processed.shape
        }
        joblib.dump(processed_data, output_path)
        if verbose:
            print(f"\n✓ Processed audio features saved to {output_path}")
    
    if verbose:
        print("\n" + "="*70)
        print("✓ Audio processing complete!")
        print("="*70)
    
    return features_processed, scaler, stats


if __name__ == "__main__":
    # Test audio processor
    from data_loader import load_gtzan_features
    
    print("Testing Audio Processor...")
    print("="*70)
    
    # Load data
    gtzan_df, feature_cols, labels = load_gtzan_features(verbose=True)
    
    # Process
    features, scaler, stats = process_audio_pipeline(
        gtzan_df, feature_cols, verbose=True
    )
    
    print("\nAudio Processor Test Complete!")
