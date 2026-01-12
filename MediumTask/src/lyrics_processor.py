"""
Lyrics Feature Processor
Extracts TF-IDF features from song lyrics
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.decomposition import TruncatedSVD # type: ignore
import re
import joblib # type: ignore
import warnings
warnings.filterwarnings('ignore')


def clean_lyrics(lyrics_text, verbose=False):
    """
    Clean lyrics text: lowercase, remove special characters, etc.
    
    Args:
        lyrics_text: Raw lyrics string
        verbose: Print information
        
    Returns:
        cleaned: Cleaned lyrics text
    """
    if pd.isna(lyrics_text):
        return ""
    
    # Convert to string and lowercase
    text = str(lyrics_text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if verbose and len(text) > 100:
        print(f"Original: {lyrics_text[:100]}...")
        print(f"Cleaned: {text[:100]}...")
    
    return text


def extract_lyrics_features(lyrics_df, 
                            max_features=5000,
                            min_df=2,
                            max_df=0.8,
                            vectorizer=None,
                            save_vectorizer=True,
                            vectorizer_path='models/tfidf_vectorizer.pkl',
                            verbose=True):
    """
    Extract TF-IDF features from lyrics.
    
    Args:
        lyrics_df: DataFrame with lyrics
        max_features: Maximum vocabulary size
        min_df: Minimum document frequency
        max_df: Maximum document frequency (fraction)
        vectorizer: Existing vectorizer (if None, creates new)
        save_vectorizer: Whether to save vectorizer
        vectorizer_path: Path to save vectorizer
        verbose: Print information
        
    Returns:
        tfidf_matrix: TF-IDF feature matrix
        vectorizer: Fitted TfidfVectorizer
        vocab_size: Actual vocabulary size
    """
    if verbose:
        print("\nExtracting lyrics features...")
        print(f"  Max features: {max_features}")
        print(f"  Min document frequency: {min_df}")
        print(f"  Max document frequency: {max_df}")
    
    # Clean all lyrics
    if verbose:
        print("\n✓ Cleaning lyrics text...")
    
    lyrics_cleaned = lyrics_df['Lyric'].apply(clean_lyrics)
    
    # Remove empty lyrics
    valid_mask = lyrics_cleaned.str.len() > 0
    lyrics_cleaned = lyrics_cleaned[valid_mask]
    
    if verbose:
        print(f"✓ Valid lyrics: {len(lyrics_cleaned)}/{len(lyrics_df)}")
        print(f"  Average lyrics length: {lyrics_cleaned.str.len().mean():.0f} chars")
    
    # Create or use vectorizer
    if vectorizer is None:
        if verbose:
            print("\n✓ Creating TF-IDF vectorizer...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            strip_accents='unicode',
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(lyrics_cleaned)
        
        if save_vectorizer:
            joblib.dump(vectorizer, vectorizer_path)
            if verbose:
                print(f"✓ Vectorizer saved to {vectorizer_path}")
    else:
        if verbose:
            print("\n✓ Using existing vectorizer...")
        tfidf_matrix = vectorizer.transform(lyrics_cleaned)
    
    # Convert to dense array
    tfidf_dense = tfidf_matrix.toarray()
    
    vocab_size = len(vectorizer.vocabulary_)
    
    if verbose:
        print(f"\n✓ TF-IDF features extracted:")
        print(f"  Matrix shape: {tfidf_dense.shape}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Sparsity: {(tfidf_dense == 0).sum() / tfidf_dense.size * 100:.2f}%")
        print(f"  Non-zero features per song (avg): {(tfidf_dense != 0).sum(axis=1).mean():.1f}")
        
        # Show top words
        feature_names = vectorizer.get_feature_names_out()
        top_indices = tfidf_dense.sum(axis=0).argsort()[-10:][::-1]
        print(f"\n  Top 10 terms by TF-IDF score:")
        for idx in top_indices:
            print(f"    - {feature_names[idx]}")
    
    return tfidf_dense, vectorizer, vocab_size


def reduce_lyrics_dimensions(tfidf_features, 
                             n_components=100,
                             svd=None,
                             save_svd=True,
                             svd_path='models/lyrics_svd.pkl',
                             verbose=True):
    """
    Reduce TF-IDF dimensions using Truncated SVD (LSA).
    
    Args:
        tfidf_features: TF-IDF matrix
        n_components: Number of dimensions to keep
        svd: Existing SVD model
        save_svd: Whether to save SVD model
        svd_path: Path to save SVD
        verbose: Print information
        
    Returns:
        features_reduced: Reduced feature matrix
        svd: Fitted SVD model
    """
    if verbose:
        print(f"\nReducing lyrics dimensions to {n_components}...")
    
    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        features_reduced = svd.fit_transform(tfidf_features)
        
        if save_svd:
            joblib.dump(svd, svd_path)
            if verbose:
                print(f"✓ SVD model saved to {svd_path}")
    else:
        features_reduced = svd.transform(tfidf_features)
    
    explained_var = svd.explained_variance_ratio_.sum()
    
    if verbose:
        print(f"✓ Dimensionality reduction complete:")
        print(f"  Original dimensions: {tfidf_features.shape[1]}")
        print(f"  Reduced dimensions: {features_reduced.shape[1]}")
        print(f"  Explained variance: {explained_var*100:.2f}%")
    
    return features_reduced, svd


def process_lyrics_pipeline(lyrics_df,
                            max_features=5000,
                            reduce_dims=True,
                            target_dims=100,
                            save_processed=True,
                            output_path='data/processed/lyrics_features.pkl',
                            verbose=True):
    """
    Complete lyrics processing pipeline.
    
    Args:
        lyrics_df: Lyrics DataFrame
        max_features: Max vocabulary for TF-IDF
        reduce_dims: Whether to apply dimensionality reduction
        target_dims: Target dimensions after reduction
        save_processed: Save processed features
        output_path: Path to save
        verbose: Print information
        
    Returns:
        features_final: Processed lyrics features
        vectorizer: TF-IDF vectorizer
        svd: SVD model (if used)
    """
    if verbose:
        print("\n" + "="*70)
        print("LYRICS PROCESSING PIPELINE")
        print("="*70)
    
    # Step 1: Extract TF-IDF features
    tfidf_features, vectorizer, vocab_size = extract_lyrics_features(
        lyrics_df,
        max_features=max_features,
        verbose=verbose
    )
    
    # Step 2: Reduce dimensions if requested
    svd = None
    if reduce_dims:
        features_final, svd = reduce_lyrics_dimensions(
            tfidf_features,
            n_components=target_dims,
            verbose=verbose
        )
    else:
        features_final = tfidf_features
    
    # Step 3: Standardize (important for VAE)
    from sklearn.preprocessing import StandardScaler # type: ignore
    scaler = StandardScaler()
    features_final = scaler.fit_transform(features_final)
    
    if verbose:
        print(f"\n✓ Final lyrics features standardized:")
        print(f"  Mean: {features_final.mean():.6f}")
        print(f"  Std: {features_final.std():.6f}")
    
    # Step 4: Save
    if save_processed:
        processed_data = {
            'features': features_final,
            'vectorizer': vectorizer,
            'svd': svd,
            'scaler': scaler,
            'vocab_size': vocab_size,
            'shape': features_final.shape
        }
        joblib.dump(processed_data, output_path)
        if verbose:
            print(f"\n✓ Processed lyrics features saved to {output_path}")
    
    if verbose:
        print("\n" + "="*70)
        print("✓ Lyrics processing complete!")
        print("="*70)
    
    return features_final, vectorizer, svd


if __name__ == "__main__":
    # Test lyrics processor
    from data_loader import load_lyrics_data
    
    print("Testing Lyrics Processor...")
    print("="*70)
    
    # Load data
    lyrics_df = load_lyrics_data(
        language_filter='en',
        max_songs=1000,
        verbose=True
    )
    
    # Process
    features, vectorizer, svd = process_lyrics_pipeline(
        lyrics_df,
        max_features=5000,
        reduce_dims=True,
        target_dims=100,
        verbose=True
    )
    
    print("\nLyrics Processor Test Complete!")
