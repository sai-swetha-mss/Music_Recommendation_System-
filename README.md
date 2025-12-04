# Music_Recommendation_System-
GROUP NUMBER 7 
Sai Swetha Manuguri - 700770652
Sai Sahith Pasumarthy - 700769267
Sai Rithvik Koragatla - 700772455
Venkata Kranti Kumar Jaddu - 700773514

# 🎵 Music Recommender System

A beautiful, AI-powered music recommendation app built with Streamlit. Discover your next favorite song through multiple recommendation methods!

## Features

### 🎵 Song-Based Recommendations
- Search and select any song from the database
- Get similar songs based on audio features
- Visual comparison of audio profiles
- Interactive radar charts for feature visualization
- Export recommendations to CSV

### 🎨 Discover by Preferences
- Filter by year range, genres, and artists
- Mood-based recommendations (Happy, Sad, Calm, Energetic)
- Customizable number of results
- Perfect for exploring new music

### 🎛️ Feature-Based Recommendations
- Adjust audio features with interactive sliders:
  - Danceability, Energy, Valence
  - Acousticness, Instrumentalness, Liveness
  - Speechiness, Tempo, Loudness
- Create custom audio profiles
- Find songs matching your exact preferences

### 🔍 Search & Explore
- Advanced search functionality
- Filter by year, artist, genre, or song name
- Sort results by multiple criteria
- Browse the entire music library

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have these files in the same directory:
   - `music_recommender.joblib` - The trained model
   - `music_recommender_model.py` - Model class definitions
   - `app.py` - Main Streamlit application

## Usage

### Option 1: Using the startup script (Recommended)

**Windows:**
```bash
run_app.bat
```

**Mac/Linux:**
```bash
chmod +x run_app.sh
./run_app.sh
```

### Option 2: Direct command

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

Press `Ctrl+C` in the terminal to stop the server.

## Model Requirements

The app expects a `music_recommender.joblib` file containing:
- `recommender`: MusicRecommender object with methods:
  - `get_similar_songs(song_id, n_recommendations)`
  - `recommend_by_preferences(min_year, max_year, genres, artists, mood, n_recommendations)`
  - `recommend_by_features(feature_vector, n_recommendations)`
- `df`: DataFrame with song information
- `audio_features`: List of feature names
- `scaler`: StandardScaler object
- `metadata`: Dictionary with training info
- `version`: Model version string

## Features Overview

### Audio Features
- **Danceability**: How suitable a track is for dancing
- **Energy**: Intensity and activity measure
- **Valence**: Musical positiveness (happiness)
- **Acousticness**: Confidence measure of acoustic sound
- **Instrumentalness**: Predicts whether a track contains vocals
- **Liveness**: Detects presence of an audience
- **Speechiness**: Detects presence of spoken words
- **Tempo**: Overall estimated tempo in BPM
- **Loudness**: Overall loudness in decibels

## Design Features

- 🎨 Dark music-themed UI with purple accents
- 📊 Interactive visualizations with Plotly
- 🎯 Responsive layout for all screen sizes
- 💾 Export functionality for all recommendation types
- 🔄 Real-time search and filtering
- 📈 Progress bars for similarity scores
- 🎵 Beautiful song cards with metadata

## Tips

1. **Song-Based**: Start with a song you love to find similar tracks
2. **Preferences**: Use mood filters to match your current vibe
3. **Feature-Based**: Experiment with sliders to discover unique combinations
4. **Search**: Use the search function to quickly find specific songs or artists

## Technology Stack

- **Streamlit**: Web app framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning backend
- **NumPy**: Numerical computations

## License

MIT License - Feel free to use and modify!

---

Made with ❤️ and 🎵
