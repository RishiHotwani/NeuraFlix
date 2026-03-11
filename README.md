# 🎬 NueralFlix – Netflix-Style Movie Recommendation System

A production-grade ML recommendation system with 3 algorithms, OMDB API integration, and a Netflix-inspired Streamlit UI.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Add Your OMDB API Key
- Get a free key at: https://www.omdbapi.com/apikey.aspx
- Enter it in the **🔑 OMDB API Key** sidebar panel
- Real movie posters will instantly load!

---

## 🤖 Three Recommendation Algorithms

### 1. 🎯 Content-Based Filtering (TF-IDF + Cosine Similarity)
**How it works:**
- Each movie is converted to a rich text "soup" combining:
  - **Genres** (weighted 3×)
  - **Director** (weighted 2×)
  - **Cast** (weighted 2×)
  - **Plot keywords** (filtered stopwords)
- Applied **TF-IDF vectorization** with bi-gram features
- **Cosine similarity** computed across all movie pairs
- Returns nearest neighbours to the selected seed movie

**Use case:** "I loved Inception, what else should I watch?"

### 2. 👥 Collaborative Filtering (SVD Matrix Factorization)
**How it works:**
- Builds a **Users × Movies** rating matrix (200 × 50)
- Normalises by subtracting per-user mean ratings
- Applies **Singular Value Decomposition** with 50 latent factors
- Reconstructs predicted ratings: `R̂[u,i] = μ[u] + U·Σ·Vᵀ`
- Returns highest predicted-rating movies the user hasn't seen

**Use case:** "What should User #3 watch next based on all users' patterns?"



---

## 📂 Project Structure

```
netflix_recommender/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
│
├── data/
│   ├── __init__.py
│   └── generate_data.py      # 50 curated movies + synthetic ratings
│
├── models/
│   ├── __init__.py
│   ├── content_based.py      # TF-IDF + Cosine Similarity
│   ├── collaborative_svd.py  # SVD Matrix Factorization
│   └── hybrid.py             # Weighted Hybrid Engine
│
└── utils/
    ├── __init__.py
    └── omdb_api.py           # OMDB API integration
```

---

## 🎨 UI Features

| Feature | Details |
|---------|---------|
| **Netflix Dark Theme** | #0d0d0d background, red accents (#e50914) |
| **Bebas Neue Font** | Cinematic hero titles |
| **Movie Cards** | Hover animations, rank badges, score badges |
| **User Profiles** | Rating history, genre preferences, similar users |
| **Analytics Dashboard** | Ratings distribution, genre analysis, SVD insights |
| **Real-Time Sliders** | Adjustable algorithm weights for Hybrid mode |
| **OMDB Poster Loading** | Real posters via API, elegant fallbacks |

---

## 📊 Dataset

- **50 curated movies** spanning Drama, Crime, Sci-Fi, Thriller, Action, and more
- **200 synthetic users** with genre-preference profiles
- **~3,000 ratings** (1.0–5.0 stars, 0.5 increments)
- Enrichable with real OMDB metadata via API key

---

## 🔑 OMDB API

Sign up free at [omdbapi.com](https://www.omdbapi.com/apikey.aspx) (1,000 req/day free tier).  
Enter your key in the sidebar for:
- 🖼️ Real movie poster images
- ⭐ IMDB ratings
- 🏆 Award information
- 🌍 Country & language data

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit |
| Content Filtering | scikit-learn TF-IDF + Cosine Similarity |
| Collaborative Filtering | scipy SVD (sparse matrix) |
| Data Processing | pandas + numpy |
| API Integration | requests + OMDB API |
| Styling | Custom CSS (Netflix-inspired) |
