"""
Hybrid Recommender - Weighted blend of Content-Based + Collaborative Filtering
Final Score = alpha * CF_score + (1 - alpha) * CB_score
"""
import pandas as pd
import numpy as np
import pickle
import os


class HybridRecommender:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.cb_model = None
        self.cf_model = None
        self.movies_df = None
        self.is_fitted = False

    def fit(self, cb_model, cf_model, movies_df: pd.DataFrame):
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.movies_df = movies_df.copy()
        self.is_fitted = True
        print(f"Hybrid Recommender ready (alpha={self.alpha})")
        return self

    def recommend(self, user_id: int, title: str, top_n: int = 10) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        cb_results = self.cb_model.get_similar_movies(title, top_n=len(self.movies_df) - 1)
        if cb_results.empty:
            return pd.DataFrame()

        cb_scores = dict(zip(cb_results["movie_id"], cb_results["similarity_score"]))
        max_cb = max(cb_scores.values()) or 1.0
        cb_scores = {k: v/max_cb for k,v in cb_scores.items()}

        cf_scores = self.cf_model.get_user_predicted_scores(user_id)
        if cf_scores:
            vals = np.array(list(cf_scores.values()))
            mn, mx = vals.min(), vals.max()
            denom = mx - mn if mx != mn else 1.0
            cf_scores = {k: (v-mn)/denom for k,v in cf_scores.items()}

        all_ids = set(cb_scores.keys()) | set(cf_scores.keys())
        records = []
        for mid in all_ids:
            cb_s = cb_scores.get(mid, 0.0)
            cf_s = cf_scores.get(mid, 0.5)
            records.append({"movie_id": mid, "hybrid_score": self.alpha*cf_s + (1-self.alpha)*cb_s,
                             "content_score": cb_s, "collab_score": cf_s})

        scores_df = pd.DataFrame(records)
        result = self.movies_df.merge(scores_df, on="movie_id", how="inner")
        result = result[result["title"].str.lower() != title.lower()]
        result = result.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)
        result["hybrid_pct"] = (result["hybrid_score"]*100).round(1)
        result["content_pct"] = (result["content_score"]*100).round(1)
        result["collab_pct"] = (result["collab_score"]*100).round(1)
        return result

    def set_alpha(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def save(self, path="models/hybrid_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path="models/hybrid_model.pkl"):
        with open(path, "rb") as f: return pickle.load(f)
