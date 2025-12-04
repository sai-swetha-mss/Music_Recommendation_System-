"""Quick test to verify the display functions work correctly"""
import pandas as pd
import joblib
from music_recommender_model import MusicRecommender

# Load original CSV
print("Loading original CSV...")
df_original = pd.read_csv('Music_Info.csv')
print(f"Loaded {len(df_original)} songs")

# Check first song
first_song = df_original.iloc[0]
print(f"\nFirst song: {first_song['name']} by {first_song['artist']}")
print(f"Danceability: {first_song['danceability']}")
print(f"Energy: {first_song['energy']}")
print(f"Valence: {first_song['valence']}")
print(f"Preview URL: {first_song.get('spotify_preview_url', 'N/A')[:50]}...")

# Load model
print("\nLoading model...")
model_data = joblib.load('music_recommender.joblib')
df_model = model_data['df']

# Check if features are scaled in model
first_song_model = df_model.iloc[0]
print(f"\nModel version of first song:")
print(f"Danceability: {first_song_model.get('danceability', 'N/A')}")
print(f"Energy: {first_song_model.get('energy', 'N/A')}")
print(f"Valence: {first_song_model.get('valence', 'N/A')}")

# Check if preview URL is preserved
print(f"Preview URL in model: {first_song_model.get('spotify_preview_url', 'N/A')[:50] if 'spotify_preview_url' in first_song_model else 'NOT FOUND'}...")

print("\n✅ Test complete!")
