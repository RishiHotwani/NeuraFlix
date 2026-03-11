"""
OMDB API Fetcher - Fetches rich movie metadata from OMDB API
"""
import requests
import pandas as pd
import numpy as np
import json
import os
import time

OMDB_API_KEY = "447afb9b"
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Curated list of popular movies across genres for a rich dataset
MOVIE_TITLES = [
    # Action/Adventure
    "The Dark Knight", "Inception", "Interstellar", "The Matrix",
    "Mad Max: Fury Road", "John Wick", "Avengers: Endgame", "Iron Man",
    "Captain America: Civil War", "Thor: Ragnarok", "Black Panther",
    "Spider-Man: Into the Spider-Verse", "Mission: Impossible – Fallout",
    "Top Gun: Maverick", "Avatar", "Gladiator", "The Bourne Identity",
    "Die Hard", "Speed", "Terminator 2: Judgment Day",

    # Sci-Fi
    "Blade Runner 2049", "Ex Machina", "Arrival", "Gravity",
    "The Martian", "2001: A Space Odyssey", "Alien", "Aliens",
    "Dune", "Ready Player One", "District 9", "Children of Men",
    "Minority Report", "Eternal Sunshine of the Spotless Mind",
    "Her", "WALL-E", "Contact", "Annihilation",

    # Drama
    "The Shawshank Redemption", "Forrest Gump", "The Godfather",
    "The Godfather Part II", "Schindler's List", "12 Angry Men",
    "The Green Mile", "A Beautiful Mind", "Good Will Hunting",
    "The Pursuit of Happyness", "Dead Poets Society", "Rain Man",
    "Million Dollar Baby", "The Revenant", "Braveheart",
    "Whiplash", "La La Land", "Moonlight", "Parasite", "Joker",

    # Thriller/Mystery
    "Silence of the Lambs", "Se7en", "Fight Club", "Gone Girl",
    "Prisoners", "Zodiac", "Knives Out", "Shutter Island",
    "The Prestige", "Memento", "Rear Window", "Vertigo",
    "Psycho", "The Sixth Sense", "Oldboy", "Parasite",

    # Comedy
    "The Grand Budapest Hotel", "Superbad", "Bridesmaids",
    "The Hangover", "Game Night", "Get Out", "Groundhog Day",
    "Home Alone", "Mrs. Doubtfire", "Ferris Bueller's Day Off",

    # Horror
    "Get Out", "A Quiet Place", "Hereditary", "The Conjuring",
    "It", "Us", "Doctor Sleep", "Bird Box", "The Witch",
    "28 Days Later", "The Babadook",

    # Romance
    "Titanic", "The Notebook", "Pride and Prejudice",
    "Before Sunrise", "Before Sunset", "Crazy Rich Asians",
    "When Harry Met Sally", "Pretty Woman",

    # Animation
    "Toy Story", "The Lion King", "Finding Nemo", "Up",
    "Coco", "Soul", "Spirited Away", "Princess Mononoke",
    "Your Name", "Howl's Moving Castle",

    # Crime
    "Pulp Fiction", "No Country for Old Men", "The Departed",
    "Heat", "City of God", "Goodfellas", "The Usual Suspects",
    "L.A. Confidential", "Reservoir Dogs", "Catch Me If You Can",

    # Fantasy
    "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers",
    "The Lord of the Rings: The Return of the King",
    "Harry Potter and the Sorcerer's Stone",
    "Harry Potter and the Prisoner of Azkaban",
    "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe",
    "Pan's Labyrinth", "Princess Bride",
]


def fetch_movie_from_omdb(title: str) -> dict | None:
    """Fetch a single movie's data from OMDB API."""
    try:
        params = {
            "apikey": OMDB_API_KEY,
            "t": title,
            "plot": "full",
            "r": "json"
        }
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        data = response.json()
        if data.get("Response") == "True":
            return data
        return None
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return None


def parse_movie_data(raw: dict) -> dict:
    """Parse raw OMDB response into clean dict."""
    def safe_float(val, default=0.0):
        try:
            return float(str(val).replace(",", "").replace("$", "").replace("N/A", "0"))
        except:
            return default

    def safe_int(val, default=0):
        try:
            return int(str(val).replace(",", "").replace("N/A", "0"))
        except:
            return default

    ratings = {r["Source"]: r["Value"] for r in raw.get("Ratings", [])}
    imdb_rating = safe_float(raw.get("imdbRating", "0"))
    rt_score = ratings.get("Rotten Tomatoes", "N/A")

    return {
        "imdb_id": raw.get("imdbID", ""),
        "title": raw.get("Title", ""),
        "year": safe_int(raw.get("Year", "0")),
        "rated": raw.get("Rated", "N/A"),
        "runtime": safe_int(str(raw.get("Runtime", "0")).replace(" min", "")),
        "genre": raw.get("Genre", ""),
        "director": raw.get("Director", ""),
        "writer": raw.get("Writer", ""),
        "actors": raw.get("Actors", ""),
        "plot": raw.get("Plot", ""),
        "language": raw.get("Language", ""),
        "country": raw.get("Country", ""),
        "awards": raw.get("Awards", ""),
        "poster": raw.get("Poster", ""),
        "imdb_rating": imdb_rating,
        "imdb_votes": safe_int(raw.get("imdbVotes", "0")),
        "metascore": safe_int(raw.get("Metascore", "0")),
        "rt_score": rt_score,
        "box_office": raw.get("BoxOffice", "N/A"),
        "production": raw.get("Production", "N/A"),
        "type": raw.get("Type", "movie"),
    }


def load_or_fetch_movies(cache_path: str = "data/movies_cache.json") -> pd.DataFrame:
    """Load movies from cache or fetch from OMDB API."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print("Loading movies from cache...")
        with open(cache_path, "r") as f:
            movies_list = json.load(f)
        df = pd.DataFrame(movies_list)
        print(f"Loaded {len(df)} movies from cache.")
        return df

    print("Fetching movies from OMDB API...")
    movies_list = []
    seen_ids = set()

    for i, title in enumerate(MOVIE_TITLES):
        print(f"Fetching ({i+1}/{len(MOVIE_TITLES)}): {title}")
        raw = fetch_movie_from_omdb(title)
        if raw:
            parsed = parse_movie_data(raw)
            if parsed["imdb_id"] and parsed["imdb_id"] not in seen_ids:
                seen_ids.add(parsed["imdb_id"])
                movies_list.append(parsed)
        time.sleep(0.2)  # Be nice to API

    with open(cache_path, "w") as f:
        json.dump(movies_list, f, indent=2)

    df = pd.DataFrame(movies_list)
    print(f"Fetched and cached {len(df)} movies.")
    return df


def get_movie_poster_url(title: str = None, imdb_id: str = None) -> str:
    """Get poster URL for a movie."""
    try:
        params = {"apikey": OMDB_API_KEY, "plot": "short", "r": "json"}
        if imdb_id:
            params["i"] = imdb_id
        elif title:
            params["t"] = title
        else:
            return ""
        response = requests.get(OMDB_BASE_URL, params=params, timeout=5)
        data = response.json()
        if data.get("Response") == "True":
            poster = data.get("Poster", "")
            if poster and poster != "N/A":
                return poster
        return ""
    except:
        return ""
