import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Import the MusicRecommender class
from music_recommender_model import MusicRecommender

# Page configuration
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark music theme with movie-style cards
st.markdown("""
<style>

    h1, h2, h3 {
        color: #bb86fc;
    }
    
    /* Movie-style card */
    .music-card {
        background: linear-gradient(135deg, #2d2d44 0%, #1f1f3a 100%);
        border-radius: 15px;
        padding: 0;
        margin: 10px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
        border: 2px solid #3a3a5a;
        height: 100%;
    }
    .music-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(187, 134, 252, 0.3);
        border-color: #bb86fc;
    }
    .card-header {
        background: linear-gradient(135deg, #bb86fc 0%, #7b4fd4 100%);
        padding: 15px;
        text-align: center;
        min-height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .card-rank {
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(0, 0, 0, 0.7);
        color: #bb86fc;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .card-body {
        padding: 15px;
    }
    .card-title {
        color: #bb86fc;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
        min-height: 50px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .card-artist {
        color: #a0a0c0;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .card-meta {
        color: #808090;
        font-size: 12px;
        margin-bottom: 10px;
    }
    .similarity-bar {
        background: #2a2a3a;
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    .similarity-fill {
        background: linear-gradient(90deg, #bb86fc 0%, #7b4fd4 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .audio-player {
        margin: 10px 0;
        width: 100%;
    }
    .feature-badge {
        display: inline-block;
        background: #3a3a5a;
        color: #bb86fc;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin: 3px;
    }
    .stButton>button {
        background-color: #bb86fc;
        color: #000;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #9d6fd4;
    }
    .play-icon {
        font-size: 40px;
        color: #bb86fc;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained music recommender model"""
    try:
        model_data = joblib.load('music_recommender.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model_data = load_model()

if model_data is None:
    st.error("Failed to load the music recommender model. Please check if 'music_recommender.joblib' exists.")
    st.stop()

# Extract components
recommender = model_data['recommender']
df = model_data['df']
audio_features = model_data['audio_features']
scaler = model_data['scaler']
metadata = model_data.get('metadata', {})

# Load original CSV for unscaled feature values
try:
    df_original = pd.read_csv('Music_Info.csv')
    # Create a mapping of track_id to original features
    original_features = {}
    for _, row in df_original.iterrows():
        track_id = row.get('track_id')
        if track_id:
            original_features[track_id] = {
                'danceability': row.get('danceability', 0.5),
                'energy': row.get('energy', 0.5),
                'valence': row.get('valence', 0.5),
                'acousticness': row.get('acousticness', 0.5),
                'tempo': row.get('tempo', 120)
            }
except:
    original_features = {}

# Helper functions
def create_radar_chart(features_dict, title="Audio Features"):
    """Create a radar chart for audio features"""
    categories = list(features_dict.keys())
    values = list(features_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(202, 240, 248, 0.3)',
        line=dict(color='#bb86fc', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(202, 240, 248,0)'
        ),
        showlegend=False,
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fff')
    )
    return fig

def display_song_card_grid(song, rank=None, similarity=None):
    """Display a song in a movie-style card format"""
    # Get preview URL
    preview_url = song.get('spotify_preview_url', '')
    
    # Get song info
    song_name = song.get('name', 'Unknown')
    song_artist = song.get('artist', 'Unknown')
    song_year = song.get('year', 'N/A')
    song_genre = str(song.get('genre', 'N/A'))[:20]
    track_id = song.get('track_id', '')
    
    # Get original unscaled feature values
    if track_id and track_id in original_features:
        orig = original_features[track_id]
        danceability = orig['danceability']
        energy = orig['energy']
        valence = orig['valence']
    else:
        # Fallback to song values (might be scaled)
        danceability = abs(song.get('danceability', 0.5))
        energy = abs(song.get('energy', 0.5))
        valence = abs(song.get('valence', 0.5))
        # Clamp to 0-1 range
        danceability = min(1.0, danceability)
        energy = min(1.0, energy)
        valence = min(1.0, valence)
    
    # Card HTML
    similarity_pct = int(similarity * 100) if similarity else 0
    rank_badge = f'<div class="card-rank">#{rank}</div>' if rank else ''
    
    card_html = f"""
    <div class="music-card">
        {rank_badge}
        <div class="card-header">
            <div class="play-icon">🎵</div>
        </div>
        <div class="card-body">
            <div class="card-title">{song_name}</div>
            <div class="card-artist">🎤 {song_artist}</div>
            <div class="card-meta">📅 {song_year} | 🎸 {song_genre}</div>
    """
    
    if similarity is not None:
        card_html += f"""
            <div class="similarity-bar">
                <div class="similarity-fill" style="width: {similarity_pct}%"></div>
            </div>
            <div style="text-align: center; color: #bb86fc; font-size: 12px; margin-top: 5px;">
                {similarity_pct}% Match
            </div>
        """
    
    card_html += """
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Audio player below the card
    if preview_url and preview_url != '' and 'http' in preview_url:
        st.audio(preview_url, format='audio/mp3')
    else:
        st.caption("🔇 No preview available")

def display_song_card(song, rank=None, similarity=None):
    """Display a song in a simple card format (for selected song)"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("🎵", unsafe_allow_html=True)
    
    with col2:
        if rank:
            st.markdown(f"### #{rank} - {song['name']}")
        else:
            st.markdown(f"### {song['name']}")
        
        st.markdown(f"**Artist:** {song['artist']}")
        st.markdown(f"**Year:** {song['year']} | **Genre:** {song.get('genre', 'N/A')}")
        
        # Audio player
        preview_url = song.get('spotify_preview_url', '')
        if preview_url and preview_url != '' and 'http' in preview_url:
            st.audio(preview_url, format='audio/mp3')
        
        if similarity is not None:
            st.progress(similarity)
            st.caption(f"Similarity: {similarity:.2%}")
        
        # Audio features - use original unscaled values
        track_id = song.get('track_id', '')
        feature_cols = st.columns(5)
        features_to_show = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
        
        for idx, feat in enumerate(features_to_show):
            # Try to get original value first
            if track_id and track_id in original_features and feat in original_features[track_id]:
                val = original_features[track_id][feat]
            elif feat in song:
                val = abs(song[feat])  # Use absolute value and clamp
                if feat != 'tempo':
                    val = min(1.0, val)
            else:
                val = 0
            
            if feat == 'tempo':
                feature_cols[idx].metric(feat.capitalize(), f"{val:.0f} BPM")
            else:
                feature_cols[idx].metric(feat.capitalize(), f"{val:.2f}")

def get_feature_comparison(song1, song2, features):
    """Compare features between two songs"""
    comparison = {}
    for feat in features:
        if feat in song1 and feat in song2:
            comparison[feat] = {
                'song1': song1[feat],
                'song2': song2[feat],
                'diff': abs(song1[feat] - song2[feat])
            }
    return comparison

# Sidebar
with st.sidebar:
    st.markdown("# 🎵 Music Recommender")
    st.markdown("---")
    
    # Model info
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Dataset Size:** {len(df):,} songs
    **Features:** {len(audio_features)} audio features
    **Training Date:** {metadata.get('training_date', 'N/A')}
    **Version:** {model_data.get('version', '1.0')}
    """)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "Select Mode:",
        ["🎵 Song-Based", "🎨 Discover by Preferences", "🎛️ Feature-Based", "🔍 Search & Explore"],
        label_visibility="collapsed"
    )

# Main content
st.markdown("# 🎵 Music Recommender System")
st.markdown("Discover your next favorite song with AI-powered recommendations")
st.markdown("---")

# Page 1: Song-Based Recommendations
if "Song-Based" in page:
    st.markdown("## 🎵 Song-Based Recommendations")
    st.markdown("Find similar songs based on a song you love")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search box
        search_query = st.text_input("🔍 Search for a song", placeholder="Enter song name or artist...")
        
        # Filter songs based on search
        if search_query:
            filtered_df = df[
                df['name'].str.contains(search_query, case=False, na=False) |
                df['artist'].str.contains(search_query, case=False, na=False)
            ].head(50)
        else:
            filtered_df = df.head(50)
        
        # Song selector
        song_options = [f"{row['name']} - {row['artist']} ({row['year']})" 
                       for _, row in filtered_df.iterrows()]
        
        if song_options:
            selected_song_str = st.selectbox("Select a song:", song_options)
            selected_idx = song_options.index(selected_song_str)
            selected_song = filtered_df.iloc[selected_idx]
            song_id = selected_song['track_id']
        else:
            st.warning("No songs found. Try a different search term.")
            st.stop()
    
    with col2:
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
    
    st.markdown("---")
    
    # Display selected song
    st.markdown("### 🎯 Selected Song")
    with st.container():
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        display_song_card(selected_song)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show audio features
    with st.expander("📊 Audio Features Visualization"):
        features_dict = {feat: selected_song[feat] for feat in audio_features 
                        if feat in selected_song and feat != 'tempo'}
        fig = create_radar_chart(features_dict, f"Audio Profile: {selected_song['name']}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Get recommendations
    if st.button("🎵 Get Recommendations", type="primary"):
        with st.spinner("Finding similar songs..."):
            try:
                recommendations = recommender.get_similar_songs(song_id, n_recommendations)
                
                st.markdown(f"### 🎼 Top {n_recommendations} Similar Songs")
                
                # Display recommendations in grid (3 columns)
                cols_per_row = 3
                for i in range(0, len(recommendations), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(recommendations):
                            rec_song = recommendations.iloc[idx]
                            with cols[j]:
                                display_song_card_grid(
                                    rec_song, 
                                    rank=idx + 1, 
                                    similarity=rec_song.get('similarity_score', 0.8)
                                )
                                
                                # Action buttons
                                btn_col1, btn_col2 = st.columns(2)
                                with btn_col1:
                                    if st.button("📊 Compare", key=f"compare_{idx}", use_container_width=True):
                                        st.session_state[f'compare_{idx}'] = True
                                with btn_col2:
                                    if st.button("🎵 More", key=f"more_{idx}", use_container_width=True):
                                        st.session_state['selected_song_id'] = rec_song['track_id']
                                        st.rerun()
                                
                                # Show comparison if button clicked
                                if st.session_state.get(f'compare_{idx}', False):
                                    with st.expander("📊 Feature Comparison", expanded=True):
                                        comparison = get_feature_comparison(selected_song, rec_song, audio_features)
                                        comp_df = pd.DataFrame(comparison).T
                                        st.dataframe(comp_df, use_container_width=True)
                
                # Export option
                if st.button("💾 Export Recommendations as CSV"):
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"recommendations_{selected_song['name']}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# Page 2: Discover by Preferences
elif "Preferences" in page:
    st.markdown("## 🎨 Discover by Preferences")
    st.markdown("Find songs that match your mood and preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Year range
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        year_range = st.slider("Year Range", min_year, max_year, (1990, 2020))
        
        # Genres
        all_genres = df['genre'].dropna().unique().tolist()
        selected_genres = st.multiselect("Genres", all_genres, default=[])
    
    with col2:
        # Artists
        all_artists = sorted(df['artist'].dropna().unique().tolist())
        search_artist = st.text_input("Search Artist", placeholder="Type to search...")
        if search_artist:
            filtered_artists = [a for a in all_artists if search_artist.lower() in a.lower()][:20]
        else:
            filtered_artists = all_artists[:20]
        selected_artists = st.multiselect("Artists", filtered_artists, default=[])
        
        # Mood
        mood = st.selectbox("Mood", ["Any", "Happy/Upbeat", "Sad/Melancholy", "Calm/Chill", "Energetic"])
        
        # Number of recommendations
        n_recommendations = st.slider("Number of recommendations", 5, 30, 15)
    
    if st.button("🎵 Discover Songs", type="primary"):
        with st.spinner("Finding songs that match your preferences..."):
            try:
                recommendations = recommender.recommend_by_preferences(
                    min_year=year_range[0],
                    max_year=year_range[1],
                    genres=selected_genres if selected_genres else None,
                    artists=selected_artists if selected_artists else None,
                    mood=mood if mood != "Any" else None,
                    n_recommendations=n_recommendations
                )
                
                if len(recommendations) == 0:
                    st.warning("No songs found matching your criteria. Try adjusting your filters.")
                else:
                    st.success(f"Found {len(recommendations)} songs!")
                    
                    # Display recommendations in grid (3 columns)
                    cols_per_row = 3
                    for i in range(0, len(recommendations), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            idx = i + j
                            if idx < len(recommendations):
                                song = recommendations.iloc[idx]
                                with cols[j]:
                                    display_song_card_grid(song, rank=idx + 1)
                    
                    # Export option
                    if st.button("💾 Export as CSV"):
                        csv = recommendations.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="discovered_songs.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error: {e}")

# Page 3: Feature-Based Recommendations
elif "Feature-Based" in page:
    st.markdown("## 🎛️ Feature-Based Recommendations")
    st.markdown("Create custom recommendations by adjusting audio features")
    
    st.markdown("### 🎚️ Adjust Audio Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
        valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5, 0.01)
    
    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2, 0.01)
    
    with col3:
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.01)
        tempo = st.slider("Tempo (BPM)", 50, 200, 120, 1)
        loudness = st.slider("Loudness (dB)", -30.0, 0.0, -5.0, 0.5)
    
    n_recommendations = st.slider("Number of recommendations", 5, 30, 10)
    
    # Create feature vector
    feature_vector = {
        'danceability': danceability,
        'energy': energy,
        'valence': valence,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'speechiness': speechiness,
        'tempo': tempo,
        'loudness': loudness
    }
    
    # Visualize target features
    with st.expander("📊 Target Audio Profile"):
        fig = create_radar_chart(
            {k: v for k, v in feature_vector.items() if k not in ['tempo', 'loudness']},
            "Target Audio Profile"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🎵 Find Matching Songs", type="primary"):
        with st.spinner("Finding songs with similar features..."):
            try:
                recommendations = recommender.recommend_by_features(feature_vector, n_recommendations)
                
                st.markdown(f"### 🎼 Top {n_recommendations} Matching Songs")
                
                # Display recommendations in grid (3 columns)
                cols_per_row = 3
                for i in range(0, len(recommendations), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(recommendations):
                            song = recommendations.iloc[idx]
                            with cols[j]:
                                display_song_card_grid(
                                    song, 
                                    rank=idx + 1, 
                                    similarity=song.get('similarity_score', 0.8)
                                )
                
                # Export option
                if st.button("💾 Export as CSV"):
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="feature_based_recommendations.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error: {e}")

# Page 4: Search & Explore
else:
    st.markdown("## 🔍 Search & Explore")
    st.markdown("Browse and search through the music library")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input("🔍 Search", placeholder="Search by song, artist, or genre...")
    
    with col2:
        search_field = st.selectbox("Search in", ["All", "Song", "Artist", "Genre"])
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Name", "Artist", "Year", "Popularity"])
    
    # Year filter
    col1, col2 = st.columns(2)
    with col1:
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        year_filter = st.slider("Filter by Year", min_year, max_year, (min_year, max_year))
    
    with col2:
        results_limit = st.slider("Results to show", 10, 100, 50)
    
    # Apply filters
    filtered_df = df[(df['year'] >= year_filter[0]) & (df['year'] <= year_filter[1])]
    
    if search_query:
        if search_field == "Song":
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_query, case=False, na=False)]
        elif search_field == "Artist":
            filtered_df = filtered_df[filtered_df['artist'].str.contains(search_query, case=False, na=False)]
        elif search_field == "Genre":
            filtered_df = filtered_df[filtered_df['genre'].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(search_query, case=False, na=False) |
                filtered_df['artist'].str.contains(search_query, case=False, na=False) |
                filtered_df['genre'].str.contains(search_query, case=False, na=False)
            ]
    
    # Sort
    if sort_by == "Name":
        filtered_df = filtered_df.sort_values('name')
    elif sort_by == "Artist":
        filtered_df = filtered_df.sort_values('artist')
    elif sort_by == "Year":
        filtered_df = filtered_df.sort_values('year', ascending=False)
    
    filtered_df = filtered_df.head(results_limit)
    
    st.markdown(f"### 📊 Found {len(filtered_df)} songs")
    
    # Display results in grid (3 columns)
    cols_per_row = 3
    for i in range(0, len(filtered_df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(filtered_df):
                song = filtered_df.iloc[idx]
                with cols[j]:
                    display_song_card_grid(song, rank=idx + 1)
                    if st.button("🎵 Get Similar", key=f"similar_{idx}"):
                        st.session_state['selected_song_id'] = song['track_id']
                        st.session_state['page'] = "Song-Based"
                        st.rerun()
    
    # Export search results
    if len(filtered_df) > 0:
        if st.button("💾 Export Search Results"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="search_results.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🎵 Music Recommender System | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
