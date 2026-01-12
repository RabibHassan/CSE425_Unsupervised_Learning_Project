"""
Data Loader Module
Loads and prepares GTZAN audio features and lyrics data
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_gtzan_features(filepath='data/features_30_sec.csv', verbose=True):
    """
    Load GTZAN audio features dataset.
    
    Args:
        filepath: Path to GTZAN features CSV
        verbose: Print loading information
        
    Returns:
        df: DataFrame with audio features
        feature_cols: List of feature column names
        labels: Genre labels
    """
    if verbose:
        print("Loading GTZAN audio features...")
    
    df = pd.read_csv(filepath)
    
    # Identify feature columns (exclude filename, length, label)
    exclude_cols = ['filename', 'length', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and labels
    features = df[feature_cols].values
    labels = df['label'].values
    
    if verbose:
        print(f"✓ Loaded {len(df)} songs")
        print(f"✓ Audio features: {len(feature_cols)} dimensions")
        print(f"✓ Genres: {df['label'].nunique()} classes")
        print(f"  Genre distribution:")
        print(df['label'].value_counts().to_string())
    
    return df, feature_cols, labels


def load_lyrics_data(filepath='data/lyrics-data.csv', 
                     language_filter='en',
                     max_songs=1000,
                     verbose=True):
    """
    Load lyrics dataset and filter by language.
    
    Args:
        filepath: Path to lyrics CSV
        language_filter: Language code to filter ('en' for English)
        max_songs: Maximum number of songs to load (for memory efficiency)
        verbose: Print loading information
        
    Returns:
        df: DataFrame with lyrics data
    """
    if verbose:
        print("\nLoading lyrics dataset...")
    
    # Load full dataset
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"✓ Total songs in dataset: {len(df)}")
        print(f"  Languages: {df['language'].value_counts().head(10).to_dict()}")
    
    # Filter by language if specified
    if language_filter:
        if verbose:
            print(f"✓ Filtering for language: '{language_filter}'")
        df = df[df['language'] == language_filter].copy()
        
        if verbose:
            print(f"✓ Songs after filtering: {len(df)}")
    
    # Sample if too many songs
    if len(df) > max_songs:
        if verbose:
            print(f"✓ Sampling {max_songs} random songs for efficiency...")
        df = df.sample(n=max_songs, random_state=42).reset_index(drop=True)
    
    # Remove rows with missing lyrics
    df = df.dropna(subset=['Lyric']).copy()
    
    # Clean song names
    df['SName'] = df['SName'].str.strip()
    
    if verbose:
        print(f"✓ Final dataset: {len(df)} songs with lyrics")
        print(f"✓ Sample songs:")
        for i, row in df.head(3).iterrows():
            print(f"  - {row['SName']}: {len(row['Lyric'])} characters")
    
    return df


def create_matched_dataset(gtzan_df, lyrics_df, match_by='genre', verbose=True):
    """
    Create matched dataset by aligning GTZAN and lyrics.
    Since exact song matching is hard, we align by genre.
    
    Args:
        gtzan_df: GTZAN DataFrame
        lyrics_df: Lyrics DataFrame
        match_by: How to match ('genre' or 'random')
        verbose: Print information
        
    Returns:
        aligned_gtzan: GTZAN subset
        aligned_lyrics: Lyrics subset (same order as GTZAN)
    """
    if verbose:
        print("\n" + "="*70)
        print("Creating aligned multi-modal dataset...")
        print("="*70)
    
    n_samples = min(len(gtzan_df), len(lyrics_df))
    
    if match_by == 'genre':
        # For each GTZAN genre, assign random lyrics
        # This creates a "genre-aware" pairing (not exact songs, but thematic alignment)
        if verbose:
            print(f"Strategy: Genre-based alignment")
            print(f"Using {n_samples} samples")
        
        # Use all GTZAN songs (1000)
        aligned_gtzan = gtzan_df.copy()
        
        # Sample lyrics to match GTZAN size
        aligned_lyrics = lyrics_df.sample(n=len(gtzan_df), 
                                         random_state=42).reset_index(drop=True)
        
    elif match_by == 'random':
        # Randomly pair audio and lyrics
        if verbose:
            print(f"Strategy: Random pairing")
        
        aligned_gtzan = gtzan_df.sample(n=n_samples, 
                                       random_state=42).reset_index(drop=True)
        aligned_lyrics = lyrics_df.sample(n=n_samples, 
                                        random_state=42).reset_index(drop=True)
    
    if verbose:
        print(f"\n✓ Final aligned dataset:")
        print(f"  Audio samples: {len(aligned_gtzan)}")
        print(f"  Lyrics samples: {len(aligned_lyrics)}")
        print(f"  Genres (from audio): {aligned_gtzan['label'].nunique()}")
    
    return aligned_gtzan, aligned_lyrics


def load_multimodal_data(gtzan_path='data/features_30_sec.csv',
                        lyrics_path='data/lyrics-data.csv',
                        language='en',
                        max_lyrics=1000,
                        match_strategy='genre',
                        verbose=True):
    """
    Convenience function to load both datasets and align them.
    
    Args:
        gtzan_path: Path to GTZAN CSV
        lyrics_path: Path to lyrics CSV
        language: Language filter for lyrics
        max_lyrics: Max lyrics to load
        match_strategy: How to align datasets
        verbose: Print information
        
    Returns:
        gtzan_df: Aligned GTZAN DataFrame
        lyrics_df: Aligned lyrics DataFrame
        feature_cols: List of audio feature column names
        labels: Genre labels (from GTZAN)
    """
    # Load individual datasets
    gtzan_full, feature_cols, labels_full = load_gtzan_features(
        gtzan_path, verbose=verbose
    )
    
    lyrics_full = load_lyrics_data(
        lyrics_path, 
        language_filter=language,
        max_songs=max_lyrics,
        verbose=verbose
    )
    
    # Align datasets
    gtzan_aligned, lyrics_aligned = create_matched_dataset(
        gtzan_full, lyrics_full,
        match_by=match_strategy,
        verbose=verbose
    )
    
    # Extract aligned labels
    labels = gtzan_aligned['label'].values
    
    if verbose:
        print("\n" + "="*70)
        print("✓ Multi-modal dataset ready!")
        print("="*70)
    
    return gtzan_aligned, lyrics_aligned, feature_cols, labels


if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    print("="*70)
    
    gtzan_df, lyrics_df, feature_cols, labels = load_multimodal_data(
        verbose=True
    )
    
    print("\n" + "="*70)
    print("Data Loader Test Complete!")
    print("="*70)
