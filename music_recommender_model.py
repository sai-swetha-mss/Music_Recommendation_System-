"""
Music Recommender Model Classes
This module contains the MusicRecommender class definition needed for loading the saved model.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class MusicRecommender:
    """
    Enhanced content-based recommender with multiple similarity options.
    """
    
    def __init__(self, df: pd.DataFrame, audio_features: List[str], 
                 use_approximate_similarity: bool = False, n_neighbors: int = 100):
        """
        Initialize recommender.
        
        Args:
            df: DataFrame with songs and features
            audio_features: List of feature column names
            use_approximate_similarity: Use kNN for large datasets (memory efficient)
            n_neighbors: Number of neighbors for approximate similarity
        """
        self.df = df.reset_index(drop=True).copy()
        self.audio_features = list(audio_features)
        self.use_approximate_similarity = use_approximate_similarity
        self.n_neighbors = n_neighbors
        
        # Ensure track_id exists
        if "track_id" not in self.df.columns:
            self.df["track_id"] = [f"track_{i:06d}" for i in range(len(self.df))]
        
        # Create mappings
        self.song_to_idx = {tid: idx for idx, tid in enumerate(self.df["track_id"].tolist())}
        self.idx_to_song = {idx: tid for tid, idx in self.song_to_idx.items()}
        
        # Feature matrix
        self.feature_matrix = self.df[self.audio_features].astype(float).values
        
        # Precompute similarity based on dataset size
        self.similarity_matrix = None
        self.knn_model = None
        
        n_songs = len(self.df)
        if use_approximate_similarity or n_songs > 10000:
            self._build_knn_model()
        else:
            self._compute_full_similarity()
    
    def _compute_full_similarity(self):
        """Compute full NxN cosine similarity matrix."""
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
    def _build_knn_model(self):
        """Build kNN model for approximate similarity."""
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.df)),
            metric='cosine',
            algorithm='auto'
        )
        self.knn_model.fit(self.feature_matrix)
    
    def get_similar_songs(self, song_id: str, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Return top-n similar songs to the provided song_id.
        """
        if song_id not in self.song_to_idx:
            # Try to find by name
            matches = self.df[self.df["name"].str.contains(song_id, case=False, na=False)]
            if not matches.empty:
                song_id = matches.iloc[0]["track_id"]
            else:
                raise KeyError(f"Song id/name '{song_id}' not found in recommender index")
        
        idx = self.song_to_idx[song_id]
        
        if self.knn_model is not None:
            # Use kNN for approximate similarity
            distances, indices = self.knn_model.kneighbors(
                self.feature_matrix[idx:idx+1], 
                n_neighbors=min(n_recommendations + 1, len(self.df))
            )
            # Convert cosine distance to similarity
            scores = 1 - distances.flatten()
            top_indices = indices.flatten()
        else:
            # Use precomputed similarity matrix
            scores = self.similarity_matrix[idx].copy()
            scores[idx] = -np.inf  # Exclude self
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            scores = scores[top_indices]
        
        # Get results
        results = self.df.iloc[top_indices].copy()
        results = results.assign(similarity_score=scores)
        
        # Remove the query song if it's in results
        results = results[results["track_id"] != song_id].head(n_recommendations)
        
        return results.reset_index(drop=True)
    
    def recommend_by_features(self, feature_vector, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Recommend songs similar to a custom feature vector.
        """
        # Handle dict input
        if isinstance(feature_vector, dict):
            # Convert dict to array matching audio_features order
            vec_list = []
            for feat in self.audio_features:
                # Try to get value from dict, use 0 if not found
                val = feature_vector.get(feat, 0)
                vec_list.append(val)
            feature_vector = vec_list
        
        if len(feature_vector) != len(self.audio_features):
            raise ValueError(f"Feature vector must have length {len(self.audio_features)}")
        
        vec = np.asarray(feature_vector).reshape(1, -1)
        
        if self.knn_model is not None:
            distances, indices = self.knn_model.kneighbors(
                vec, 
                n_neighbors=min(n_recommendations, len(self.df))
            )
            scores = 1 - distances.flatten()
            top_indices = indices.flatten()
        else:
            sims = cosine_similarity(vec, self.feature_matrix)[0]
            top_indices = np.argsort(sims)[::-1][:n_recommendations]
            scores = sims[top_indices]
        
        results = self.df.iloc[top_indices].copy()
        results = results.assign(similarity_score=scores)
        return results.reset_index(drop=True)
    
    def recommend_by_preferences(self,
                                 min_year: Optional[int] = None,
                                 max_year: Optional[int] = None,
                                 genres: Optional[List[str]] = None,
                                 artists: Optional[List[str]] = None,
                                 mood: Optional[str] = None,
                                 n_recommendations: int = 20) -> pd.DataFrame:
        """
        Filter by metadata preferences and return recommendations.
        """
        filtered = self.df.copy()
        
        # Apply filters
        if min_year is not None and "year" in filtered.columns:
            filtered = filtered[filtered["year"] >= min_year]
        
        if max_year is not None and "year" in filtered.columns:
            filtered = filtered[filtered["year"] <= max_year]
        
        if genres:
            pattern = "|".join([f"(?i){g.strip()}" for g in genres if g.strip()])
            if pattern:
                genre_mask = filtered["genre"].astype(str).str.contains(pattern, na=False)
                tags_mask = filtered.get("tags", "").astype(str).str.contains(pattern, na=False)
                filtered = filtered[genre_mask | tags_mask]
        
        if artists:
            pattern = "|".join([f"(?i){a.strip()}" for a in artists if a.strip()])
            if pattern:
                filtered = filtered[filtered["artist"].astype(str).str.contains(pattern, na=False)]
        
        # Apply mood filter if specified
        if mood and "valence" in self.audio_features:
            mood_lower = mood.lower()
            if "happy" in mood_lower or "upbeat" in mood_lower:
                filtered = filtered[(filtered["valence"] > 0.6) & (filtered["energy"] > 0.6)]
            elif "sad" in mood_lower or "melancholy" in mood_lower:
                filtered = filtered[(filtered["valence"] < 0.4) & (filtered["energy"] < 0.5)]
            elif "calm" in mood_lower or "chill" in mood_lower:
                if "energy" in self.audio_features:
                    filtered = filtered[filtered["energy"] < 0.5]
            elif "energetic" in mood_lower:
                if "energy" in self.audio_features:
                    filtered = filtered[filtered["energy"] > 0.7]
        
        if filtered.empty:
            return self.df.sample(min(n_recommendations, len(self.df))).reset_index(drop=True)
        
        # If filtered set is small, return random from it
        if len(filtered) <= n_recommendations:
            return filtered.sample(min(n_recommendations, len(filtered))).reset_index(drop=True)
        
        # Otherwise, compute similarity to centroid of filtered set
        centroid = filtered[self.audio_features].mean(axis=0).values.reshape(1, -1)
        
        if self.knn_model is not None:
            distances, indices = self.knn_model.kneighbors(
                centroid, 
                n_neighbors=min(n_recommendations * 2, len(self.df))
            )
            scores = 1 - distances.flatten()
            candidate_indices = indices.flatten()
        else:
            sims = cosine_similarity(centroid, self.feature_matrix)[0]
            candidate_indices = np.argsort(sims)[::-1][:n_recommendations * 2]
            scores = sims[candidate_indices]
        
        # Filter to only include songs that pass the original filters
        filtered_indices = set(filtered.index)
        valid_mask = [idx in filtered_indices for idx in candidate_indices]
        
        if not any(valid_mask):
            return filtered.sample(min(n_recommendations, len(filtered))).reset_index(drop=True)
        
        valid_indices = candidate_indices[valid_mask][:n_recommendations]
        valid_scores = scores[valid_mask][:n_recommendations]
        
        results = self.df.iloc[valid_indices].copy()
        results = results.assign(similarity_score=valid_scores)
        return results.reset_index(drop=True)
    
    def search_songs(self, query: str, search_fields: List[str] = None, n_results: int = 10) -> pd.DataFrame:
        """
        Search songs by name, artist, or genre.
        """
        if search_fields is None:
            search_fields = ["name", "artist", "genre"]
        
        query_lower = query.lower()
        masks = []
        
        for field in search_fields:
            if field in self.df.columns:
                mask = self.df[field].astype(str).str.lower().str.contains(query_lower, na=False)
                masks.append(mask)
        
        if not masks:
            return pd.DataFrame()
        
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask | mask
        
        results = self.df[combined_mask].head(n_results).copy()
        return results
