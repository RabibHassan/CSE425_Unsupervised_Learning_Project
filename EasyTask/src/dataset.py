"""
Dataset Processing Utilities
Functions for loading and preprocessing music dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath):
    """
    Load hybrid language music dataset.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with music features
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} songs, {df.shape[1]} columns")
    return df


def extract_features(df, exclude_cols=['filename', 'label', 'language']):
    """
    Extract feature matrix from dataset.

    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude from features

    Returns:
        X: Feature matrix
        y_language: Language labels
        y_genre: Genre labels
        feature_names: List of feature column names
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y_language = df['language'].values if 'language' in df.columns else None
    y_genre = df['label'].values if 'label' in df.columns else None

    print(f"Features extracted: {len(feature_cols)} features")

    return X, y_language, y_genre, feature_cols


def preprocess_features(X, scaler=None):
    """
    Standardize features to zero mean and unit variance.

    Args:
        X: Feature matrix
        scaler: Pre-fitted scaler (optional)

    Returns:
        X_scaled: Standardized features
        scaler: Fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    print(f"Features standardized: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")

    return X_scaled, scaler


def get_dataset_info(df):
    """
    Print dataset information.

    Args:
        df: Input dataframe
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total samples: {len(df)}")

    if 'language' in df.columns:
        print(f"\nLanguage distribution:")
        print(df['language'].value_counts())

    if 'label' in df.columns:
        print(f"\nGenre distribution:")
        print(df['label'].value_counts())

    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print("="*60 + "\n")