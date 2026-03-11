"""
OMDB API Integration for fetching real movie posters and metadata.
"""

import requests
import os

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "YOUR_API_KEY_HERE")
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Fallback poster URL (Netflix-style placeholder)
FALLBACK_POSTER = "https://via.placeholder.com/300x450/141414/e50914?text=No+Poster"

_poster_cache = {}


def fetch_movie_details(title: str, year: int = None) -> dict:
    """Fetch movie details from OMDB API."""
    if OMDB_API_KEY == "YOUR_API_KEY_HERE":
        return {}
    
    cache_key = f"{title}_{year}"
    if cache_key in _poster_cache:
        return _poster_cache[cache_key]
    
    params = {
        "apikey": OMDB_API_KEY,
        "t": title,
        "type": "movie",
    }
    if year:
        params["y"] = year
    
    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=5)
        data = response.json()
        if data.get("Response") == "True":
            _poster_cache[cache_key] = data
            return data
    except Exception:
        pass
    
    return {}


def get_poster_url(title: str, year: int = None) -> str:
    """Get poster URL for a movie title."""
    details = fetch_movie_details(title, year)
    poster = details.get("Poster", "")
    if poster and poster != "N/A":
        return poster
    return generate_placeholder_poster(title)


def generate_placeholder_poster(title: str) -> str:
    """Generate a colored placeholder poster URL."""
    colors = [
        ("1a1a2e", "e50914"),
        ("16213e", "f5c518"),
        ("0f3460", "e94560"),
        ("533483", "ffffff"),
        ("2b2d42", "ef233c"),
        ("1b1b2f", "e43f5a"),
        ("162447", "e43f5a"),
        ("1f4068", "1b262c"),
    ]
    color_idx = sum(ord(c) for c in title) % len(colors)
    bg, fg = colors[color_idx]
    text = title[:15].replace(" ", "+")
    return f"https://via.placeholder.com/300x450/{bg}/{fg}?text={text}"


def get_imdb_rating(title: str, year: int = None) -> str:
    """Get IMDB rating for a movie."""
    details = fetch_movie_details(title, year)
    return details.get("imdbRating", "N/A")


def get_movie_metadata(title: str, year: int = None) -> dict:
    """Get enriched metadata: poster, rating, runtime, etc."""
    details = fetch_movie_details(title, year)
    return {
        "poster": get_poster_url(title, year),
        "imdb_rating": details.get("imdbRating", "N/A"),
        "runtime": details.get("Runtime", "N/A"),
        "rated": details.get("Rated", "N/A"),
        "awards": details.get("Awards", "N/A"),
        "box_office": details.get("BoxOffice", "N/A"),
        "language": details.get("Language", "N/A"),
        "country": details.get("Country", "N/A"),
    }


def set_api_key(key: str):
    """Set the OMDB API key at runtime."""
    global OMDB_API_KEY
    OMDB_API_KEY = key
