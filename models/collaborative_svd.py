"""
Collaborative Filtering using SVD (Matrix Factorization).
Predicts what a specific user would rate/enjoy based on patterns
across all users' ratings.
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


class CollaborativeFilteringSVD:
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        self.ratings_matrix = None
        self.ratings_matrix_norm = None
        self.user_ratings_mean = None
        self.predicted_ratings = None
        self.user_ids = None
        self.movie_ids = None
        self.movies_df = None
        self.ratings_df = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Fit SVD on user-movie ratings matrix.
        ratings_df: columns [userId, movieId, rating]
        movies_df: columns [id, title, ...]
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()

        # Create user-movie pivot table
        self.ratings_matrix = ratings_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        self.user_ids = list(self.ratings_matrix.index)
        self.movie_ids = list(self.ratings_matrix.columns)

        # Normalize: subtract user mean rating
        R = self.ratings_matrix.values
        self.user_ratings_mean = np.mean(R, axis=1)
        self.ratings_matrix_norm = R - self.user_ratings_mean.reshape(-1, 1)

        # Apply SVD
        k = min(self.n_factors, min(self.ratings_matrix_norm.shape) - 1)
        U, sigma, Vt = svds(self.ratings_matrix_norm, k=k)

        self.user_factors = U
        self.sigma = np.diag(sigma)
        self.item_factors = Vt

        # Reconstruct predicted ratings
        self.predicted_ratings = pd.DataFrame(
            np.dot(np.dot(U, self.sigma), Vt) + self.user_ratings_mean.reshape(-1, 1),
            columns=self.ratings_matrix.columns,
            index=self.ratings_matrix.index
        )

        return self

    def recommend_for_user(self, user_id: int, n: int = 10, only_unseen: bool = True) -> pd.DataFrame:
        """
        Recommend top N movies for a given user.
        If only_unseen=True, exclude movies already rated by the user.
        """
        if user_id not in self.user_ids:
            return pd.DataFrame()

        # Get predicted ratings for this user
        user_preds = self.predicted_ratings.loc[user_id].copy()

        if only_unseen:
            # Get movies already rated by this user
            seen_movies = self.ratings_df[
                self.ratings_df["userId"] == user_id
            ]["movieId"].values
            # Zero out already seen
            user_preds = user_preds.drop(
                labels=[m for m in seen_movies if m in user_preds.index],
                errors="ignore"
            )

        # Sort by predicted rating
        top_movie_ids = user_preds.nlargest(n).index.tolist()
        top_scores = user_preds.nlargest(n).values

        # Merge with movie info
        result = self.movies_df[self.movies_df["id"].isin(top_movie_ids)].copy()

        # Add predicted rating
        score_map = dict(zip(top_movie_ids, top_scores))
        result["predicted_rating"] = result["id"].map(score_map)
        result = result.sort_values("predicted_rating", ascending=False)

        return result.reset_index(drop=True)

    def get_user_profile(self, user_id: int) -> dict:
        """Get a user's rating history and profile."""
        if user_id not in self.user_ids:
            return {}

        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id].copy()
        user_ratings = user_ratings.merge(
            self.movies_df[["id", "title", "genres"]],
            left_on="movieId",
            right_on="id",
            how="left"
        )

        return {
            "total_rated": len(user_ratings),
            "avg_rating": round(user_ratings["rating"].mean(), 2),
            "top_rated": user_ratings.nlargest(5, "rating")[["title", "rating"]].to_dict("records"),
            "genre_preferences": self._get_genre_preferences(user_ratings),
        }

    def _get_genre_preferences(self, user_ratings: pd.DataFrame) -> dict:
        """Compute genre preferences from user ratings."""
        genre_scores = {}
        for _, row in user_ratings.iterrows():
            if pd.isna(row.get("genres")):
                continue
            genres = str(row["genres"]).split()
            for genre in genres:
                if genre not in genre_scores:
                    genre_scores[genre] = []
                genre_scores[genre].append(row["rating"])

        return {
            genre: round(np.mean(scores), 2)
            for genre, scores in sorted(
                genre_scores.items(),
                key=lambda x: np.mean(x[1]),
                reverse=True
            )
        }

    def get_user_similarity(self, user_id: int, top_n: int = 5) -> list:
        """Find most similar users based on rating patterns."""
        if user_id not in self.user_ids:
            return []

        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_factors[user_idx]

        similarities = []
        for i, uid in enumerate(self.user_ids):
            if uid == user_id:
                continue
            sim = np.dot(user_vector, self.user_factors[i]) / (
                np.linalg.norm(user_vector) * np.linalg.norm(self.user_factors[i]) + 1e-8
            )
            similarities.append((uid, float(sim)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    def get_all_user_ids(self) -> list:
        """Return all user IDs."""
        return self.user_ids

    def get_rating_matrix_stats(self) -> dict:
        """Get statistics about the ratings matrix."""
        total_cells = self.ratings_matrix.shape[0] * self.ratings_matrix.shape[1]
        filled_cells = (self.ratings_matrix != 0).sum().sum()
        return {
            "n_users": self.ratings_matrix.shape[0],
            "n_movies": self.ratings_matrix.shape[1],
            "n_ratings": filled_cells,
            "sparsity": round(1 - filled_cells / total_cells, 4),
            "density": round(filled_cells / total_cells, 4),
        }
