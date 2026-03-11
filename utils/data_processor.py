"""
Data Preprocessor - Cleans and prepares movie data + generates synthetic ratings
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich movie dataframe."""
    df = df.copy()
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset=["imdb_id"]).reset_index(drop=True)

    # Fill nulls
    df["genre"] = df["genre"].fillna("")
    df["director"] = df["director"].fillna("")
    df["actors"] = df["actors"].fillna("")
    df["plot"] = df["plot"].fillna("")
    df["language"] = df["language"].fillna("")
    df["imdb_rating"] = df["imdb_rating"].fillna(0.0)
    df["year"] = df["year"].fillna(0).astype(int)
    df["runtime"] = df["runtime"].fillna(0).astype(int)
    df["imdb_votes"] = df["imdb_votes"].fillna(0).astype(int)

    # Create combined text soup for content-based filtering
    df["genre_clean"] = df["genre"].apply(lambda x: x.replace(", ", " ").replace("-", ""))
    df["director_clean"] = df["director"].apply(lambda x: " ".join(x.split(",")[:1]).strip())
    df["actors_clean"] = df["actors"].apply(
        lambda x: " ".join([a.strip().replace(" ", "") for a in x.split(",")[:3]])
    )
    df["soup"] = (
        df["genre_clean"] + " " +
        df["director_clean"] + " " +
        df["actors_clean"] + " " +
        df["plot"].apply(lambda x: x[:200]) + " " +
        df["genre_clean"]  # weight genre more
    )

    # Normalize ratings for hybrid
    scaler = MinMaxScaler()
    valid_ratings = df["imdb_rating"] > 0
    if valid_ratings.sum() > 0:
        df.loc[valid_ratings, "norm_rating"] = scaler.fit_transform(
            df.loc[valid_ratings, ["imdb_rating"]]
        )
    df["norm_rating"] = df["norm_rating"].fillna(0.5)

    # Assign integer movie_id for collaborative filtering
    df["movie_id"] = range(len(df))

    return df


def generate_synthetic_ratings(movies_df: pd.DataFrame,
                                 n_users: int = 50,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic user-movie ratings for collaborative filtering.
    Users have genre preferences baked in.
    """
    np.random.seed(seed)
    n_movies = len(movies_df)

    # Define user archetypes by genre preference
    genre_list = ["Action", "Drama", "Comedy", "Thriller", "Sci-Fi",
                  "Romance", "Horror", "Animation", "Crime", "Fantasy", "Adventure"]

    records = []
    for user_id in range(1, n_users + 1):
        # Each user prefers 2-3 genres
        fav_genres = np.random.choice(genre_list, size=np.random.randint(2, 4), replace=False)

        # Rate 15-40 movies per user (capped at dataset size)
        max_rate = len(movies_df)
        n_rated = min(np.random.randint(15, 41), max_rate)
        rated_movies = np.random.choice(movies_df["movie_id"].values, size=n_rated, replace=False)

        for movie_id in rated_movies:
            movie = movies_df[movies_df["movie_id"] == movie_id].iloc[0]
            movie_genres = movie["genre"]

            # Base rating from IMDB (scaled to 1-5)
            base = (movie["imdb_rating"] / 10.0) * 4 + 1 if movie["imdb_rating"] > 0 else 3.0

            # Boost if movie matches user's fav genres
            genre_boost = sum(1 for g in fav_genres if g.lower() in movie_genres.lower()) * 0.5

            # Add noise
            noise = np.random.normal(0, 0.5)
            rating = np.clip(base + genre_boost + noise, 1.0, 5.0)
            rating = round(rating * 2) / 2  # Round to nearest 0.5

            records.append({
                "user_id": user_id,
                "movie_id": int(movie_id),
                "rating": rating
            })

    ratings_df = pd.DataFrame(records)
    return ratings_df


def get_genre_distribution(movies_df: pd.DataFrame) -> dict:
    """Count movies per genre for analytics."""
    genre_counts = {}
    for genres in movies_df["genre"]:
        for g in str(genres).split(","):
            g = g.strip()
            if g and g != "N/A":
                genre_counts[g] = genre_counts.get(g, 0) + 1
    return dict(sorted(genre_counts.items(), key=lambda x: -x[1]))


def get_decade_distribution(movies_df: pd.DataFrame) -> dict:
    """Count movies per decade."""
    decade_counts = {}
    for year in movies_df["year"]:
        if year and year > 1900:
            decade = (year // 10) * 10
            label = f"{decade}s"
            decade_counts[label] = decade_counts.get(label, 0) + 1
    return dict(sorted(decade_counts.items()))
