import joblib
from music_recommender_model import MusicRecommender

print("Loading model...")
model_data = joblib.load('music_recommender.joblib')

print("Model loaded successfully!")
print(f"Dataset size: {len(model_data['df'])} songs")
print(f"Audio features: {len(model_data['audio_features'])} features")
print(f"Version: {model_data.get('version', 'N/A')}")

# Test a recommendation
recommender = model_data['recommender']
print(f"\nRecommender type: {type(recommender)}")
print("Testing get_similar_songs method...")

# Get first song
first_song_id = model_data['df'].iloc[0]['track_id']
print(f"First song: {model_data['df'].iloc[0]['name']} by {model_data['df'].iloc[0]['artist']}")

try:
    recs = recommender.get_similar_songs(first_song_id, n_recommendations=5)
    print(f"\nFound {len(recs)} recommendations!")
    print("\nTop 3 recommendations:")
    for i, row in recs.head(3).iterrows():
        print(f"  {i+1}. {row['name']} - {row['artist']}")
except Exception as e:
    print(f"Error: {e}")
