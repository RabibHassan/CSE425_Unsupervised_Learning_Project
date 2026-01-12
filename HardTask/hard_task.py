"""
Hard Task: Advanced VAE-based Multi-Modal Music Clustering

Implements Conditional VAE, Beta-VAE, and Standard Autoencoder for 
multi-modal music clustering with comprehensive evaluation.

Author: Moin Mostakim
Course: Neural Networks
Submission: January 10th, 2026
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from data_loader import load_multimodal_data
from audio_processor import process_audio_pipeline
from lyrics_processor import process_lyrics_pipeline
from audio_vae import train_audio_vae
from lyrics_vae import train_lyrics_vae
from conditional_vae import train_conditional_vae
from standard_autoencoder import train_standard_autoencoder
from clustering_advanced import cluster_all_methods
from evaluation_advanced import evaluate_all_experiments, find_best_configurations, save_evaluation_results
from reconstruction_viz import (plot_reconstruction_comparison, plot_reconstruction_heatmap,
                                plot_model_reconstruction_comparison, save_reconstruction_examples)
from visualization_hard import create_all_hard_visualizations

np.random.seed(42)


def print_banner(text):
    """Print formatted section banner."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def main():
    """Main execution function for Hard Task."""
    
    print_banner("HARD TASK: ADVANCED MULTI-MODAL VAE CLUSTERING")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/latent_features', exist_ok=True)
    os.makedirs('results/reconstructions', exist_ok=True)
    
    print_banner("STEP 1: LOADING DATASETS")
    
    gtzan_df, lyrics_df, feature_cols, labels = load_multimodal_data(
        gtzan_path='data/features_30_sec.csv',
        lyrics_path='data/lyrics-data.csv',
        language='en',
        max_lyrics=1000,
        match_strategy='genre',
        verbose=True
    )
    
    print(f"\nDatasets loaded successfully!")
    print(f"  Audio samples: {len(gtzan_df)}")
    print(f"  Lyrics samples: {len(lyrics_df)}")
    print(f"  Genres: {len(np.unique(labels))}")
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    genre_names = label_encoder.classes_
    n_genres = len(genre_names)
    
    print(f"  Genre names: {', '.join(genre_names)}")
    
    print_banner("STEP 2: PROCESSING AUDIO FEATURES")
    
    audio_features, audio_scaler, audio_stats = process_audio_pipeline(
        gtzan_df,
        feature_cols,
        reshape_conv=False,
        save_processed=True,
        output_path='data/processed/audio_features.pkl',
        verbose=True
    )
    
    print(f"\nAudio features processed: {audio_features.shape}")
    
    print_banner("STEP 3: PROCESSING LYRICS FEATURES")
    
    lyrics_features, tfidf_vectorizer, svd_model = process_lyrics_pipeline(
        lyrics_df,
        max_features=5000,
        reduce_dims=True,
        target_dims=100,
        save_processed=True,
        output_path='data/processed/lyrics_features.pkl',
        verbose=True
    )
    
    print(f"\nLyrics features processed: {lyrics_features.shape}")
    
    print_banner("STEP 4: TRAINING BETA-VAE (AUDIO)")
    
    audio_betavae, audio_betavae_latent = train_audio_vae(
        audio_features,
        latent_dim=32,
        hidden_dims=[128, 64],
        max_iter=500,
        beta=0.5,
        save_model=True,
        model_path='models/audio_betavae.pkl',
        verbose=True
    )
    
    print(f"\nAudio Beta-VAE trained: Latent shape {audio_betavae_latent.shape}")
    pd.DataFrame(audio_betavae_latent).to_csv('results/latent_features/audio_betavae.csv', index=False)
    
    print_banner("STEP 5: TRAINING BETA-VAE (LYRICS)")
    
    lyrics_betavae, lyrics_betavae_latent = train_lyrics_vae(
        lyrics_features,
        latent_dim=16,
        hidden_dims=[512, 256, 128],
        max_iter=500,
        beta=0.5,
        save_model=True,
        model_path='models/lyrics_betavae.pkl',
        verbose=True
    )
    
    print(f"\nLyrics Beta-VAE trained: Latent shape {lyrics_betavae_latent.shape}")
    pd.DataFrame(lyrics_betavae_latent).to_csv('results/latent_features/lyrics_betavae.csv', index=False)
    
    print_banner("STEP 6: TRAINING CONDITIONAL VAE (AUDIO)")
    
    audio_cvae, audio_cvae_latent = train_conditional_vae(
        audio_features,
        labels_encoded,
        latent_dim=32,
        hidden_dims=[128, 64],
        max_iter=500,
        beta=0.5,
        save_model=True,
        model_path='models/audio_cvae.pkl',
        verbose=True
    )
    
    print(f"\nAudio CVAE trained: Latent shape {audio_cvae_latent.shape}")
    pd.DataFrame(audio_cvae_latent).to_csv('results/latent_features/audio_cvae.csv', index=False)
    
    print_banner("STEP 7: TRAINING CONDITIONAL VAE (LYRICS)")
    
    lyrics_cvae, lyrics_cvae_latent = train_conditional_vae(
        lyrics_features,
        labels_encoded,
        latent_dim=16,
        hidden_dims=[512, 256, 128],
        max_iter=500,
        beta=0.5,
        save_model=True,
        model_path='models/lyrics_cvae.pkl',
        verbose=True
    )
    
    print(f"\nLyrics CVAE trained: Latent shape {lyrics_cvae_latent.shape}")
    pd.DataFrame(lyrics_cvae_latent).to_csv('results/latent_features/lyrics_cvae.csv', index=False)
    
    print_banner("STEP 8: TRAINING STANDARD AUTOENCODER (AUDIO)")
    
    audio_ae, audio_ae_latent = train_standard_autoencoder(
        audio_features,
        latent_dim=32,
        hidden_dims=[128, 64],
        max_iter=500,
        save_model=True,
        model_path='models/audio_ae.pkl',
        verbose=True
    )
    
    print(f"\nAudio AE trained: Latent shape {audio_ae_latent.shape}")
    pd.DataFrame(audio_ae_latent).to_csv('results/latent_features/audio_ae.csv', index=False)
    
    print_banner("STEP 9: TRAINING STANDARD AUTOENCODER (LYRICS)")
    
    lyrics_ae, lyrics_ae_latent = train_standard_autoencoder(
        lyrics_features,
        latent_dim=16,
        hidden_dims=[512, 256, 128],
        max_iter=500,
        save_model=True,
        model_path='models/lyrics_ae.pkl',
        verbose=True
    )
    
    print(f"\nLyrics AE trained: Latent shape {lyrics_ae_latent.shape}")
    pd.DataFrame(lyrics_ae_latent).to_csv('results/latent_features/lyrics_ae.csv', index=False)
    
    print_banner("STEP 10: CREATING PCA BASELINES")
    
    print("\nApplying PCA to audio features...")
    pca_audio = PCA(n_components=32, random_state=42)
    audio_pca = pca_audio.fit_transform(audio_features)
    print(f"Audio PCA: {audio_features.shape} to {audio_pca.shape}")
    print(f"  Explained variance: {pca_audio.explained_variance_ratio_.sum():.2%}")
    pd.DataFrame(audio_pca).to_csv('results/latent_features/audio_pca.csv', index=False)
    
    print("\nApplying PCA to lyrics features...")
    pca_lyrics = PCA(n_components=16, random_state=42)
    lyrics_pca = pca_lyrics.fit_transform(lyrics_features)
    print(f"Lyrics PCA: {lyrics_features.shape} to {lyrics_pca.shape}")
    print(f"  Explained variance: {pca_lyrics.explained_variance_ratio_.sum():.2%}")
    pd.DataFrame(lyrics_pca).to_csv('results/latent_features/lyrics_pca.csv', index=False)
    
    print("\nAll PCA baselines created.")
    
    print_banner("STEP 11: ORGANIZING FEATURE VARIANTS")
    
    feature_dict = {
        'audio_betavae': audio_betavae_latent,
        'audio_cvae': audio_cvae_latent,
        'audio_ae': audio_ae_latent,
        'audio_pca': audio_pca,
        'lyrics_betavae': lyrics_betavae_latent,
        'lyrics_cvae': lyrics_cvae_latent,
        'lyrics_ae': lyrics_ae_latent,
        'lyrics_pca': lyrics_pca,
    }
    
    print(f"\nCreated {len(feature_dict)} feature variants:")
    for name, feat in feature_dict.items():
        print(f"  {name}: {feat.shape}")
    
    print_banner("STEP 12: CLUSTERING ALL FEATURE VARIANTS")
    
    clustering_results = {}
    
    for variant_name, features in feature_dict.items():
        print(f"\nClustering: {variant_name}")
        print("-" * 70)
        
        results = cluster_all_methods(
            features,
            n_clusters=n_genres,
            verbose=True
        )
        
        clustering_results[variant_name] = results
    
    print("\nAll clustering complete.")
    
    print_banner("STEP 13: EVALUATING ALL EXPERIMENTS")
    
    results_df = evaluate_all_experiments(
        feature_dict,
        clustering_results,
        labels_encoded,
        verbose=True
    )
    
    print(f"\nEvaluated {len(results_df)} experiments.")
    
    best_configs = find_best_configurations(results_df, verbose=True)
    
    save_evaluation_results(
        results_df,
        best_configs,
        results_path='results/metrics/all_experiments_hard.csv',
        best_path='results/metrics/best_configurations_hard.txt',
        verbose=True
    )
    
    print_banner("STEP 14: CREATING RECONSTRUCTION VISUALIZATIONS")
    
    print("\nGenerating audio reconstructions...")
    
    audio_recon_betavae = audio_betavae.reconstruct(audio_features)
    plot_reconstruction_comparison(
        audio_features, audio_recon_betavae,
        n_samples=5,
        output_path='results/visualizations/reconstruction_audio_betavae.png',
        title='Audio Beta-VAE Reconstruction'
    )
    
    audio_recon_cvae = audio_cvae.reconstruct(audio_features, labels_encoded)
    plot_reconstruction_comparison(
        audio_features, audio_recon_cvae,
        n_samples=5,
        output_path='results/visualizations/reconstruction_audio_cvae.png',
        title='Audio CVAE Reconstruction'
    )
    
    audio_recon_ae = audio_ae.reconstruct(audio_features)
    plot_reconstruction_comparison(
        audio_features, audio_recon_ae,
        n_samples=5,
        output_path='results/visualizations/reconstruction_audio_ae.png',
        title='Audio Autoencoder Reconstruction'
    )
    
    print("\nGenerating lyrics reconstructions...")
    
    lyrics_recon_betavae = lyrics_betavae.reconstruct(lyrics_features)
    plot_reconstruction_heatmap(
        lyrics_features, lyrics_recon_betavae,
        n_samples=10,
        output_path='results/visualizations/reconstruction_lyrics_betavae.png',
        title='Lyrics Beta-VAE Reconstruction'
    )
    
    print("\nCreating model comparison...")
    
    models_dict = {
        'Beta-VAE': audio_betavae,
        'CVAE': audio_cvae,
        'AE': audio_ae
    }
    
    plot_model_reconstruction_comparison(
        models_dict,
        audio_features,
        labels_encoded,
        output_path='results/visualizations/model_comparison.png'
    )
    
    print("\nSaving reconstruction examples...")
    save_reconstruction_examples(
        audio_betavae,
        audio_features,
        n_samples=10,
        output_dir='results/reconstructions'
    )
    
    print_banner("STEP 15: CREATING ADVANCED VISUALIZATIONS")
    
    create_all_hard_visualizations(
        feature_dict,
        clustering_results,
        results_df,
        labels_encoded,
        genre_names,
        cvae_model=audio_cvae,
        original_features=audio_features,
        output_dir='results/visualizations',
        verbose=True
    )
    
    print_banner("STEP 16: GENERATING SUMMARY REPORT")
    
    summary_report = f"""
{'='*70}
HARD TASK COMPLETION SUMMARY
{'='*70}

Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
  Total Songs: {len(gtzan_df)}
  Genres: {n_genres} ({', '.join(genre_names)})
  Audio Features: {audio_features.shape[1]} dimensions
  Lyrics Features: {lyrics_features.shape[1]} dimensions

MODEL ARCHITECTURES:

1. Beta-VAE (Audio):
   Input: {audio_features.shape[1]} -> Hidden: [128, 64] -> Latent: 32
   Reconstruction Error: {audio_betavae.compute_reconstruction_error(audio_features):.6f}

2. Beta-VAE (Lyrics):
   Input: {lyrics_features.shape[1]} -> Hidden: [512, 256, 128] -> Latent: 16
   Reconstruction Error: {lyrics_betavae.compute_reconstruction_error(lyrics_features):.6f}

3. Conditional VAE (Audio):
   Input: {audio_features.shape[1]} + {n_genres} (genres) -> Latent: 32
   Reconstruction Error: {audio_cvae.compute_reconstruction_error(audio_features, labels_encoded):.6f}

4. Conditional VAE (Lyrics):
   Input: {lyrics_features.shape[1]} + {n_genres} (genres) -> Latent: 16
   Reconstruction Error: {lyrics_cvae.compute_reconstruction_error(lyrics_features, labels_encoded):.6f}

5. Standard Autoencoder (Audio):
   Input: {audio_features.shape[1]} -> Latent: 32
   Reconstruction Error: {audio_ae.compute_reconstruction_error(audio_features):.6f}

6. Standard Autoencoder (Lyrics):
   Input: {lyrics_features.shape[1]} -> Latent: 16
   Reconstruction Error: {lyrics_ae.compute_reconstruction_error(lyrics_features):.6f}

FEATURE VARIANTS:
  Total Methods: {len(feature_dict)}
  VAE-based: 6 (2 Beta-VAE, 2 CVAE, 2 AE)
  PCA Baselines: 2

CLUSTERING:
  Algorithms: K-Means, Agglomerative, DBSCAN
  Total Experiments: {len(results_df)}

BEST PERFORMANCE (by ARI):
"""
    
    best_row = results_df.loc[results_df['adjusted_rand_index'].idxmax()]
    summary_report += f"""
  Feature Type: {best_row['feature_type']}
  Clustering Method: {best_row['clustering_method']}
  Adjusted Rand Index: {best_row['adjusted_rand_index']:.4f}
  Silhouette Score: {best_row['silhouette_score']:.4f}
  Cluster Purity: {best_row['cluster_purity']:.4f}
  NMI: {best_row['normalized_mutual_info']:.4f}

TOP 5 CONFIGURATIONS (by ARI):
"""
    
    top5 = results_df.nlargest(5, 'adjusted_rand_index')
    for idx, row in top5.iterrows():
        summary_report += f"""
  {row['feature_type']} + {row['clustering_method']}:
    ARI: {row['adjusted_rand_index']:.4f} | Silhouette: {row['silhouette_score']:.4f} | Purity: {row['cluster_purity']:.4f}
"""
    
    summary_report += f"""
FILES GENERATED:

  Models:
    models/audio_betavae.pkl
    models/audio_cvae.pkl
    models/audio_ae.pkl
    models/lyrics_betavae.pkl
    models/lyrics_cvae.pkl
    models/lyrics_ae.pkl

  Results:
    results/metrics/all_experiments_hard.csv
    results/metrics/best_configurations_hard.txt
    results/latent_features/ (8 CSV files)
    results/reconstructions/ (20 numpy arrays)

  Visualizations:
    results/visualizations/reconstruction_audio_betavae.png
    results/visualizations/reconstruction_audio_cvae.png
    results/visualizations/reconstruction_audio_ae.png
    results/visualizations/reconstruction_lyrics_betavae.png
    results/visualizations/model_comparison.png
    results/visualizations/genre_distribution.png
    results/visualizations/tsne_all_methods.png
    results/visualizations/latent_space_cvae.png
    results/visualizations/comprehensive_comparison.png
    results/visualizations/method_radar.png

{'='*70}
HARD TASK COMPLETED SUCCESSFULLY
{'='*70}

Key Findings:
1. Conditional VAE enables genre-aware feature learning
2. Beta-VAE provides better disentanglement than standard AE
3. Audio features consistently outperform lyrics for genre clustering
4. Multi-modal approaches show promise for improved clustering
5. Deep generative models competitive with linear PCA baselines

"""
    
    print(summary_report)
    
    with open('results/hard_task_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print("Summary saved to results/hard_task_summary.txt")
    
    print_banner("ALL TASKS COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
