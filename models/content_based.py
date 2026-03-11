"""
Content-Based Filtering Model
TF-IDF + Cosine Similarity
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2), min_df=1)
        self.cosine_sim_matrix = None
        self.movies_df = None
        self.title_to_idx = {}
        self.is_fitted = False

    def fit(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df["soup"].fillna(""))
        self.cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.title_to_idx = {t.lower(): i for i, t in enumerate(self.movies_df["title"])}
        self.is_fitted = True
        print(f"ContentBased fitted on {len(self.movies_df)} movies.")
        return self

    def _resolve_title(self, title: str):
        tl = title.lower()
        if tl in self.title_to_idx:
            return tl
        matches = [t for t in self.title_to_idx if tl in t]
        return matches[0] if matches else None

    def get_similar_movies(self, title: str, top_n: int = 10, exclude_same_director: bool = False) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        tl = self._resolve_title(title)
        if tl is None:
            return pd.DataFrame()
        idx = self.title_to_idx[tl]
        sims = list(enumerate(self.cosine_sim_matrix[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        sims = [s for s in sims if s[0] != idx]
        if exclude_same_director:
            qd = self.movies_df.iloc[idx]["director"]
            sims = [s for s in sims if self.movies_df.iloc[s[0]]["director"] != qd]
        top = sims[:top_n]
        result = self.movies_df.iloc[[i for i,_ in top]].copy()
        result["similarity_score"] = [s for _,s in top]
        result["similarity_pct"] = (result["similarity_score"] * 100).round(1)
        return result.reset_index(drop=True)

    def get_recommendations_by_genre(self, genre: str, top_n: int = 10) -> pd.DataFrame:
        mask = self.movies_df["genre"].str.contains(genre, case=False, na=False)
        return self.movies_df[mask].sort_values("imdb_rating", ascending=False).head(top_n).reset_index(drop=True)

    def get_content_features(self, title: str) -> dict:
        tl = self._resolve_title(title)
        if tl is None:
            return {}
        m = self.movies_df.iloc[self.title_to_idx[tl]]
        return {"genres": m["genre"], "director": m["director"], "actors": m["actors"]}

    def save(self, path="models/content_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path="models/content_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
