import pandas as pd # pyright: ignore[reportMissingModuleSource]

# Check GTZAN features
print("="*70)
print("GTZAN AUDIO FEATURES")
print("="*70)
gtzan = pd.read_csv('data/features_30_sec.csv')
print(f"Shape: {gtzan.shape}")
print(f"\nColumns ({len(gtzan.columns)} total):")
print(gtzan.columns.tolist())
print(f"\nFirst row:")
print(gtzan.iloc[0])
print(f"\nGenre distribution:")
if 'label' in gtzan.columns:
    print(gtzan['label'].value_counts())
elif 'genre' in gtzan.columns:
    print(gtzan['genre'].value_counts())

print("\n" + "="*70)
print("LYRICS DATA")
print("="*70)
lyrics = pd.read_csv('data/lyrics-data.csv') 
print(f"Shape: {lyrics.shape}")
print(f"\nColumns ({len(lyrics.columns)} total):")
print(lyrics.columns.tolist())
print(f"\nFirst 3 rows:")
print(lyrics.head(3))
