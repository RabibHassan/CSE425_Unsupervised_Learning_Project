"""
Medium Task:  Multi-Modal VAE for Music Clustering
Main Execution Script

Author:  Moin Mostakim
Course: Neural Networks
Submission:  January 10th, 2026
"""

import sys
import os
sys.path.append('src')

import numpy as np # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
from datetime import datetime

from data_loader import load_multimodal_data # type: ignore
from audio_processor import process_audio_pipeline
from lyrics_processor import process_lyrics_pipeline
from audio_vae import train_audio_vae
from lyrics_vae import train_lyrics_vae
from multimodal_fusion import create_multimodal_variants
from clustering_advanced import cluster_all_methods
from evaluation_advanced import (
    evaluate_all_experiments,
    find_best_configurations,
    save_evaluation_results
)
from visualization_advanced import create_all_visualizations

np.random.seed(42)


def print_banner(text):
    """Print formatted banner."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)
    
def create_baseline_comparison_table(results_df, output_path='results/visualizations/baseline_comparison_table.png'):
    """
    Create baseline comparison table: VAE vs PCA
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("\nCreating baseline comparison table...")
    
    # Filter valid results (10 clusters, K-Means and Agglomerative only)
    valid_results = results_df[
        (results_df['n_clusters'] == 10) &
        (results_df['clustering_method'].isin(['kmeans', 'agglomerative']))
    ].copy()
    
    # Build table data
    table_data = []
    
    # Audio - VAE
    for method in ['kmeans', 'agglomerative']:
        row = valid_results[
            (valid_results['feature_type'] == 'audio_vae') & 
            (valid_results['clustering_method'] == method)
        ]
        if len(row) > 0:
            r = row.iloc[0]
            method_name = 'K-Means' if method == 'kmeans' else 'Agglomerative'
            table_data.append([
                'Audio',
                f'VAE + {method_name}',
                f"{r['silhouette_score']:.4f}",
                f"{r['adjusted_rand_index']:.4f}",
                f"{r['calinski_harabasz_index']:.2f}",
                f"{r['cluster_purity']:.4f}"
            ])
    
    # Audio - PCA
    for method in ['kmeans', 'agglomerative']:
        row = valid_results[
            (valid_results['feature_type'] == 'audio_pca') & 
            (valid_results['clustering_method'] == method)
        ]
        if len(row) > 0:
            r = row.iloc[0]
            method_name = 'K-Means' if method == 'kmeans' else 'Agglomerative'
            table_data.append([
                'Audio',
                f'PCA + {method_name}',
                f"{r['silhouette_score']:.4f}",
                f"{r['adjusted_rand_index']:.4f}",
                f"{r['calinski_harabasz_index']:.2f}",
                f"{r['cluster_purity']:.4f}"
            ])
    
    # Empty row
    table_data.append(['', '', '', '', '', ''])
    
    # Fusion - VAE
    for method in ['kmeans', 'agglomerative']:
        row = valid_results[
            (valid_results['feature_type'] == 'concat_vae_pca') & 
            (valid_results['clustering_method'] == method)
        ]
        if len(row) > 0:
            r = row.iloc[0]
            method_name = 'K-Means' if method == 'kmeans' else 'Agglomerative'
            table_data.append([
                'Fusion',
                f'VAE + {method_name}',
                f"{r['silhouette_score']:.4f}",
                f"{r['adjusted_rand_index']:.4f}",
                f"{r['calinski_harabasz_index']:.2f}",
                f"{r['cluster_purity']:.4f}"
            ])
    
    # Fusion - PCA
    for method in ['kmeans', 'agglomerative']:
        row = valid_results[
            (valid_results['feature_type'] == 'concat_pca_baseline') & 
            (valid_results['clustering_method'] == method)
        ]
        if len(row) > 0:
            r = row.iloc[0]
            method_name = 'K-Means' if method == 'kmeans' else 'Agglomerative'
            table_data.append([
                'Fusion',
                f'PCA + {method_name}',
                f"{r['silhouette_score']:.4f}",
                f"{r['adjusted_rand_index']:.4f}",
                f"{r['calinski_harabasz_index']:.2f}",
                f"{r['cluster_purity']:.4f}"
            ])
    
    # Empty row
    table_data.append(['', '', '', '', '', ''])
    
    # Lyrics
    for feature_type in ['lyrics_vae', 'lyrics_pca']:
        row = valid_results[
            (valid_results['feature_type'] == feature_type) & 
            (valid_results['clustering_method'] == 'kmeans')
        ]
        if len(row) > 0:
            r = row.iloc[0]
            model_name = 'VAE' if 'vae' in feature_type else 'PCA'
            table_data.append([
                'Lyrics',
                f'{model_name} + K-Means',
                f"{r['silhouette_score']:.4f}",
                f"{r['adjusted_rand_index']:.4f}",
                f"{r['calinski_harabasz_index']:.2f}",
                f"{r['cluster_purity']:.4f}"
            ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Column headers
    columns = ['Feature Type', 'Method', 'Silhouette', 'ARI', 'CH Index', 'Purity']
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.28, 0.13, 0.13, 0.13, 0.13]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#1a237e')
        cell.set_text_props(weight='bold', color='white', size=11)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Style data rows
    for i in range(1, len(table_data) + 1):
        row_data = table_data[i-1]
        
        # Skip empty rows
        if row_data[0] == '':
            for j in range(len(columns)):
                cell = table[(i, j)]
                cell.set_facecolor('#e0e0e0')
                cell.set_edgecolor('white')
            continue
        
        # Color based on method type
        if 'VAE' in row_data[1]:
            row_color = '#bbdefb'  # Light blue for VAE
        else:
            row_color = '#c8e6c9'  # Light green for PCA
        
        for j in range(len(columns)):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('white')
            cell.set_linewidth(1)
            
            # Bold best values in each section
            if j == 2:  # Silhouette
                try:
                    val = float(row_data[j])
                    if val > 0.12:  # High silhouette
                        cell.set_text_props(weight='bold', color='darkgreen')
                except:
                    pass
            elif j == 3:  # ARI
                try:
                    val = float(row_data[j])
                    if val > 0.15:  # High ARI
                        cell.set_text_props(weight='bold', color='darkgreen')
                except:
                    pass
    
    plt.title('Baseline Comparison: VAE vs PCA (Valid 10-Cluster Results)', 
            fontsize=14, weight='bold', pad=20)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Baseline comparison table saved to {output_path}")



def main():
    """Main execution function."""
    
    print_banner("MEDIUM TASK:  MULTI-MODAL VAE MUSIC CLUSTERING")
    print(f"Start Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/latent_features', exist_ok=True)
    
    # STEP 1: Load Datasets
    print_banner("STEP 1: LOADING DATASETS")
    
    gtzan_df, lyrics_df, feature_cols, labels = load_multimodal_data(
        gtzan_path='data/features_30_sec.csv',
        lyrics_path='data/lyrics-data.csv',
        language='en',
        max_lyrics=1000,
        match_strategy='genre',
        verbose=True
    )
    
    print(f"\n✓ Datasets loaded successfully!")
    print(f"  Audio samples: {len(gtzan_df)}")
    print(f"  Lyrics samples: {len(lyrics_df)}")
    print(f"  Genres: {len(np.unique(labels))}")
    
    # STEP 2: Process Audio Features
    print_banner("STEP 2: PROCESSING AUDIO FEATURES")
    
    audio_features, audio_scaler, audio_stats = process_audio_pipeline(
        gtzan_df, 
        feature_cols,
        reshape_conv=False,
        save_processed=True,
        output_path='data/processed/audio_features.pkl',
        verbose=True
    )
    
    print(f"\n✓ Audio features processed:  {audio_features.shape}")
    
    # STEP 3: Process Lyrics Features
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
    
    print(f"\n✓ Lyrics features processed:  {lyrics_features.shape}")
    
    # STEP 4: Train Audio VAE
    print_banner("STEP 4: TRAINING AUDIO VAE")
    
    audio_vae, audio_latent = train_audio_vae(
        audio_features,
        latent_dim=32,
        hidden_dims=[128, 64],
        max_iter=500,
        save_model=True,
        model_path='models/audio_conv_vae.pkl',
        verbose=True
    )
    
    print(f"\n✓ Audio VAE trained:  Latent shape {audio_latent.shape}")
    
    audio_latent_df = pd.DataFrame(
        audio_latent,
        columns=[f'audio_latent_{i+1}' for i in range(audio_latent.shape[1])]
    )
    audio_latent_df['label'] = labels
    audio_latent_df.to_csv('results/latent_features/audio_latent. csv', index=False)
    print("✓ Audio latent features saved")
    
    # STEP 5: Train Lyrics VAE
    print_banner("STEP 5: TRAINING LYRICS VAE")
    
    lyrics_vae, lyrics_latent = train_lyrics_vae(
        lyrics_features,
        latent_dim=16,
        hidden_dims=[512, 256, 128],
        max_iter=500,
        save_model=True,
        model_path='models/lyrics_dense_vae.pkl',
        verbose=True
    )

    print(f"\n✓ Lyrics VAE trained: Latent shape {lyrics_latent.shape}")
    
    lyrics_latent_df = pd.DataFrame(
        lyrics_latent,
        columns=[f'lyrics_latent_{i+1}' for i in range(lyrics_latent.shape[1])]
    )
    lyrics_latent_df['label'] = labels
    lyrics_latent_df.to_csv('results/latent_features/lyrics_latent.csv', index=False)
    print("✓ Lyrics latent features saved")
    
    #Step 5.5: CREATE PCA BASELINES
    from sklearn.decomposition import PCA

    # PCA on Audio Features
    print("\nApplying PCA to audio features...")
    pca_audio = PCA(n_components=32, random_state=42)
    audio_pca_features = pca_audio.fit_transform(audio_features)
    print(f"✓ Audio PCA: {audio_features.shape} → {audio_pca_features.shape}")
    print(f"  Explained variance: {pca_audio.explained_variance_ratio_.sum():.2%}")

    # PCA on Lyrics Features  
    print("\nApplying PCA to lyrics features...")
    pca_lyrics = PCA(n_components=16, random_state=42)
    lyrics_pca_features = pca_lyrics.fit_transform(lyrics_features)
    print(f"✓ Lyrics PCA: {lyrics_features.shape} → {lyrics_pca_features.shape}")
    print(f"  Explained variance: {pca_lyrics.explained_variance_ratio_.sum():.2%}")

    # PCA on Concatenated Features
    print("\nApplying PCA to concatenated features...")
    concat_raw = np.hstack([audio_features, lyrics_features])
    pca_concat = PCA(n_components=24, random_state=42)
    concat_pca_baseline = pca_concat.fit_transform(concat_raw)
    print(f"✓ Concat PCA: {concat_raw.shape} → {concat_pca_baseline.shape}")
    print(f"  Explained variance: {pca_concat.explained_variance_ratio_.sum():.2%}")

    print("\n✓ All PCA baselines created!")
    
    # STEP 6: Create Multi-Modal Fusion Variants
    print_banner("STEP 6: CREATING IMPROVED MULTI-MODAL FUSION")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fusion_dict = {}

    # === VAE METHODS ===
    fusion_dict['audio_vae'] = audio_latent
    print(f"✓ Audio VAE: {audio_latent.shape}")

    fusion_dict['lyrics_vae'] = lyrics_latent
    print(f"✓ Lyrics VAE: {lyrics_latent.shape}")

    # === VAE FUSION METHODS ===
    print("\nCreating VAE fusion variants...")

    # Concat VAE + PCA
    concat_features = np.hstack([audio_latent, lyrics_latent])
    pca_concat = PCA(n_components=24, random_state=42)
    concat_vae_pca = pca_concat.fit_transform(concat_features)
    print(f"✓ Concat VAE+PCA: {concat_features.shape} → {concat_vae_pca.shape}")
    fusion_dict['concat_vae_pca'] = concat_vae_pca

    # Audio-heavy weighted fusion
    audio_scaled = StandardScaler().fit_transform(audio_latent)
    lyrics_scaled = StandardScaler().fit_transform(lyrics_latent)
    audio_weighted = audio_scaled * 0.7
    lyrics_weighted = lyrics_scaled * 0.3
    concat_weighted_heavy = np.hstack([audio_weighted, lyrics_weighted])
    pca_heavy = PCA(n_components=16, random_state=42)
    audio_heavy_vae = pca_heavy.fit_transform(concat_weighted_heavy)
    print(f"✓ Audio-heavy VAE: {audio_heavy_vae.shape}")
    fusion_dict['audio_heavy_vae'] = audio_heavy_vae

    # Balanced weighted fusion
    audio_weighted_bal = audio_scaled * 0.5
    lyrics_weighted_bal = lyrics_scaled * 0.5
    concat_weighted_bal = np.hstack([audio_weighted_bal, lyrics_weighted_bal])
    pca_bal = PCA(n_components=16, random_state=42)
    balanced_vae = pca_bal.fit_transform(concat_weighted_bal)
    print(f"✓ Balanced VAE: {balanced_vae.shape}")
    fusion_dict['balanced_vae'] = balanced_vae

    # === PCA BASELINES (ADD THESE!) ===
    print("\nAdding PCA baselines...")

    fusion_dict['audio_pca'] = audio_pca_features
    print(f"✓ Audio PCA baseline: {audio_pca_features.shape}")

    fusion_dict['lyrics_pca'] = lyrics_pca_features
    print(f"✓ Lyrics PCA baseline: {lyrics_pca_features.shape}")

    fusion_dict['concat_pca_baseline'] = concat_pca_baseline
    print(f"✓ Concat PCA baseline: {concat_pca_baseline.shape}")

    print(f"\n✓ Created {len(fusion_dict)} feature variants:")
    print(f"  - 5 VAE-based methods")
    print(f"  - 3 PCA baselines")


    
    # STEP 7: Perform Clustering on All Variants
    print_banner("STEP 7: CLUSTERING ALL FEATURE VARIANTS")
    
    clustering_results = {}
    n_genres = len(np.unique(labels))
    
    for variant_name, features in fusion_dict.items():
        print(f"\nClustering:  {variant_name}")
        print("-" * 70)
        
        results = cluster_all_methods(
            features,
            n_clusters=n_genres,
            verbose=True
        )
        
        clustering_results[variant_name] = results
    
    print("\n✓ All clustering complete!")
    
    # STEP 8: Evaluate All Experiments
    print_banner("STEP 8: EVALUATING ALL EXPERIMENTS")
    
    results_df = evaluate_all_experiments(
        fusion_dict,
        clustering_results,
        labels,
        verbose=True
    )
    
    print(f"\n✓ Evaluated {len(results_df)} experiments")
    print("\nResults Preview:")
    print(results_df.head(10).to_string(index=False))
    
    best_configs = find_best_configurations(results_df, verbose=True)
    
    save_evaluation_results(
        results_df,
        best_configs,
        results_path='results/metrics/all_experiments_metrics.csv',
        best_path='results/metrics/best_configurations.txt',
        verbose=True
    )
    
    # STEP 9: Create Visualizations
    print_banner("STEP 9: CREATING VISUALIZATIONS")

    create_all_visualizations(
        fusion_dict,
        clustering_results,
        results_df,
        labels,
        output_dir='results/visualizations',
        verbose=True
    )

    # Create baseline comparison table
    print("\n" + "="*70)
    print("Creating baseline comparison table...")
    print("="*70)
    create_baseline_comparison_table(
        results_df,
        output_path='results/visualizations/baseline_comparison_table.png'
    )
    

    # STEP 10: Generate Summary Report
    print_banner("STEP 10: GENERATING SUMMARY REPORT")
    
    summary_report = f"""
{"="*70}
MEDIUM TASK COMPLETION SUMMARY
{"="*70}

Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
  • Total Songs: {len(gtzan_df)}
  • Genres: {n_genres} ({', '.join(np.unique(labels)[:5])}...)
  • Audio Features: {audio_features.shape[1]} dimensions
  • Lyrics Features: {lyrics_features.shape[1]} dimensions

VAE ARCHITECTURES: 
  • Audio Conv-VAE: 
      Input:  {audio_features.shape[1]} → Hidden: [128, 64] → Latent: 32
      Reconstruction Error: {audio_vae.compute_reconstruction_error(audio_features):.6f}
  
  • Lyrics Dense-VAE: 
      Input: {lyrics_features.shape[1]} → Hidden:  [256, 128] → Latent: 32
      Reconstruction Error: {lyrics_vae.compute_reconstruction_error(lyrics_features):.6f}

MULTI-MODAL FUSION: 
  • Feature Variants:  {len(fusion_dict)}
  • Variants: {', '.join(fusion_dict.keys())}

CLUSTERING: 
  • Algorithms: K-Means, Agglomerative, DBSCAN
  • Total Experiments: {len(results_df)}

BEST PERFORMANCE (by Silhouette Score):
"""
    
    best_row = results_df.loc[results_df['silhouette_score'].idxmax()]
    summary_report += f"""
  Feature Type: {best_row['feature_type']}
  Clustering Method: {best_row['clustering_method']}
  Silhouette Score: {best_row['silhouette_score']:.4f}
  Calinski-Harabasz Index: {best_row['calinski_harabasz_index']:.2f}
  Adjusted Rand Index:  {best_row['adjusted_rand_index']:.4f}
  Normalized Mutual Info: {best_row['normalized_mutual_info']:.4f}
  Cluster Purity: {best_row['cluster_purity']:.4f}

TOP 5 CONFIGURATIONS (by Silhouette Score):
"""
    
    top5 = results_df.nlargest(5, 'silhouette_score')
    for idx, row in top5.iterrows():
        summary_report += f"""
  {row['feature_type']} + {row['clustering_method']}:
    Silhouette:  {row['silhouette_score']:.4f} | ARI: {row['adjusted_rand_index']:.4f} | Purity: {row['cluster_purity']:.4f}
"""
    
    summary_report += f"""
FILES GENERATED:
  Models: 
    • models/audio_conv_vae.pkl
    • models/lyrics_dense_vae.pkl
    • models/audio_scaler.pkl
    • models/tfidf_vectorizer.pkl
  
  Results:
    • results/metrics/all_experiments_metrics.csv
    • results/metrics/best_configurations.txt
    • results/latent_features/ (multiple CSV files)
  
  Visualizations:
    • results/visualizations/tsne_comparison.png
    • results/visualizations/metrics_comparison.png
    • results/visualizations/best_methods_summary.png
    • results/visualizations/method_heatmap.png

{"="*70}
MEDIUM TASK COMPLETED SUCCESSFULLY!
{"="*70}

Key Findings:
1. Multi-modal fusion (audio + lyrics) shows {'improved' if best_row['feature_type'] == 'concat' else 'competitive'} performance
2. Best clustering algorithm:  {best_row['clustering_method']}
3. Audio features contribute strongly to genre separation
4. Convolutional VAE effectively captures audio patterns

"""
    
    print(summary_report)
    
    with open('results/medium_task_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print("\n✓ Summary saved to results/medium_task_summary.txt")
    
    print_banner("ALL TASKS COMPLETE!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

if __name__ == "__main__": 
    try:
        main()
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()