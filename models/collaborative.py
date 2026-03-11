"""
Collaborative Filtering - SVD Matrix Factorization
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pickle
import os


class CollaborativeFilteringRecommender:
    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors
        self.user_factors = None
        self.sigma = None
        self.movie_factors = None
        self.predicted_ratings = None
        self.user_ids = []
        self.movie_ids = []
        self.ratings_matrix = None
        self.movies_df = None
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy()
        pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
        self.user_ids = list(pivot.index)
        self.movie_ids = list(pivot.columns)
        self.ratings_matrix = pivot.values.astype(float)

        user_mean = np.mean(self.ratings_matrix, axis=1, keepdims=True)
        demeaned = self.ratings_matrix - user_mean
        sparse = csr_matrix(demeaned)
        k = min(self.n_factors, min(sparse.shape) - 1)
        U, sigma, Vt = svds(sparse, k=k)
        idx = np.argsort(-sigma)
        U, sigma, Vt = U[:,idx], sigma[idx], Vt[idx,:]
        self.user_factors, self.sigma, self.movie_factors = U, sigma, Vt
        self.predicted_ratings = np.clip(np.dot(np.dot(U, np.diag(sigma)), Vt) + user_mean, 1.0, 5.0)
        self.is_fitted = True
        print(f"CF SVD fitted: {len(self.user_ids)} users, {len(self.movie_ids)} movies, k={k}")
        return self

    def recommend_for_user(self, user_id: int, top_n: int = 10, only_unrated: bool = True) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        if user_id not in self.user_ids:
            return self.movies_df.nlargest(top_n, "imdb_rating")
        ui = self.user_ids.index(user_id)
        predicted = self.predicted_ratings[ui].copy()
        if only_unrated:
            predicted[self.ratings_matrix[ui] > 0] = -np.inf
        top_idx = np.argsort(-predicted)[:top_n]
        top_mids = [self.movie_ids[i] for i in top_idx if predicted[i] > -np.inf]
        top_scores = [predicted[i] for i in top_idx if predicted[i] > -np.inf]
        result = self.movies_df[self.movies_df["movie_id"].isin(top_mids)].copy()
        result["predicted_rating"] = result["movie_id"].map(dict(zip(top_mids, top_scores)))
        return result.sort_values("predicted_rating", ascending=False).head(top_n).reset_index(drop=True)

    def get_user_history(self, user_id: int, ratings_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        ur = ratings_df[ratings_df["user_id"] == user_id].sort_values("rating", ascending=False)
        return ur.merge(self.movies_df, on="movie_id", how="left").head(top_n)[["title","rating","genre","imdb_rating","poster"]]

    def get_user_predicted_scores(self, user_id: int) -> dict:
        if not self.is_fitted or user_id not in self.user_ids:
            return {}
        ui = self.user_ids.index(user_id)
        return dict(zip(self.movie_ids, self.predicted_ratings[ui]))

    def get_latent_factors_info(self) -> dict:
        if not self.is_fitted:
            return {}
        ev = (self.sigma**2) / np.sum(self.sigma**2)
        return {"n_factors": len(self.sigma), "top_singular_values": self.sigma[:5].tolist(),
                "explained_variance_ratio": ev[:5].tolist(), "total_variance_explained": float(np.sum(ev))}

    def save(self, path="models/collab_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path="models/collab_model.pkl"):
        with open(path, "rb") as f: return pickle.load(f)
