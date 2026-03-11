"""
NeuralFlix - Netflix-Style Movie Recommendation System
Built with Streamlit | OMDB API | TF-IDF | SVD
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.omdb_fetcher import load_or_fetch_movies
from utils.data_processor import (
    preprocess_movies, generate_synthetic_ratings,
    get_genre_distribution, get_decade_distribution
)
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender

# ──────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralFlix — AI Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────
#  NETFLIX DARK THEME CSS
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --red: #E50914; --red-dark: #B20710; --gold: #F5C518;
    --dark: #0A0A0A; --dark2: #141414; --dark3: #1C1C1C;
    --dark4: #252525; --text: #E5E5E5; --muted: #8C8C8C;
    --card: #1A1A1A; --border: #2A2A2A;
}

/* ── Global reset ── */
html, body { background-color: var(--dark2) !important; }
.stApp { background-color: var(--dark2) !important; }
[class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 1.5rem 2rem 4rem 2rem !important; max-width: 1400px !important; }

/* ── Sidebar – target every known Streamlit Cloud selector ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child,
div[data-testid="stSidebarContent"] {
    background-color: var(--dark) !important;
    background: var(--dark) !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border) !important;
}

/* Sidebar radio button nav styling */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    color: #8C8C8C !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    display: block !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #1C1C1C !important;
    color: #E5E5E5 !important;
}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio label[data-active="true"] {
    background: #1C1C1C !important;
    color: var(--text) !important;
    border-left: 3px solid var(--red) !important;
}
/* Hide the default radio circle dots */
section[data-testid="stSidebar"] .stRadio [role="radio"] {
    display: none !important;
}
section[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
}

/* ── Main content text ── */
.stApp, .stApp p, .stApp div, .stApp span {
    color: var(--text);
}

/* ── Logo ── */
.logo-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem; letter-spacing: 0.1em;
    background: linear-gradient(135deg, #E50914, #FF4444, #FFAA00);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.logo-sub { font-size: 0.7rem; letter-spacing: 0.22em; color: var(--muted); text-transform: uppercase; }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0505 60%, #0d0d0d 100%);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}

/* ── Section header ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.55rem; letter-spacing: 0.08em;
    color: var(--text); border-left: 4px solid var(--red);
    padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}
.section-badge {
    display: inline-block; background: var(--red); color: white;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 2px 7px;
    border-radius: 3px; margin-left: 0.4rem; vertical-align: middle;
}

/* ── Movie card ── */
.movie-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; transition: all 0.22s ease;
}
.movie-card:hover {
    border-color: var(--red); transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(229,9,20,0.2);
}
.score-badge {
    position: absolute; top: 8px; right: 8px;
    background: rgba(0,0,0,0.85); border: 1px solid var(--gold);
    color: var(--gold); font-size: 0.68rem; font-weight: 700;
    padding: 3px 7px; border-radius: 4px;
}
.match-badge {
    position: absolute; top: 8px; left: 8px;
    background: rgba(229,9,20,0.9); color: white;
    font-size: 0.63rem; font-weight: 700; padding: 3px 7px; border-radius: 4px;
}

/* ── Stat card ── */
.stat-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1.2rem; text-align:center; }
.stat-value { font-family:'Bebas Neue',sans-serif; font-size:2.2rem; color:var(--red); line-height:1; }
.stat-label { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:3px; }

/* ── Algo card ── */
.algo-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1.2rem; height:100%; position:relative; overflow:hidden; }
.algo-card::after { content:''; position:absolute; top:0;left:0;right:0; height:3px; background:linear-gradient(90deg,var(--red),var(--gold)); }
.algo-icon { font-size:1.8rem; margin-bottom:0.4rem; }
.algo-title { font-weight:700; font-size:0.9rem; margin-bottom:0.3rem; }
.algo-desc { font-size:0.76rem; color:var(--muted); line-height:1.5; }
.algo-tech { display:inline-block; background:var(--dark4); border:1px solid var(--red); color:var(--red); font-size:0.62rem; font-weight:600; padding:2px 7px; border-radius:3px; margin-top:7px; }

/* ── Info panel ── */
.info-panel { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1.4rem; margin-bottom:1rem; }

/* ── Metrics ── */
div[data-testid="stMetric"] { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:8px !important; padding:0.7rem 1rem !important; }
div[data-testid="stMetric"] label { color:var(--muted) !important; font-size:0.75rem !important; }
div[data-testid="stMetricValue"] { color:var(--red) !important; font-family:'Bebas Neue',sans-serif !important; font-size:1.8rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; gap:0 !important; border-bottom:1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important; border:none !important; padding:0.55rem 1.4rem !important; font-weight:500 !important; font-size:0.88rem !important; }
.stTabs [aria-selected="true"] { color:var(--text) !important; border-bottom:2px solid var(--red) !important; }

/* ── Buttons — main content area ── */
.block-container .stButton > button {
    background: var(--red) !important; color: white !important; border: none !important;
    border-radius: 6px !important; font-weight: 600 !important; font-size: 0.84rem !important;
    letter-spacing: 0.04em !important; padding: 0.5rem 1.4rem !important;
    transition: background 0.2s !important; width: 100% !important;
}
.block-container .stButton > button:hover { background: var(--red-dark) !important; }

/* ── Sidebar nav buttons — inactive ── */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #8C8C8C !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    text-align: left !important;
    padding: 7px 12px !important;
    width: 100% !important;
    transition: all 0.15s !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1C1C1C !important;
    color: #E5E5E5 !important;
}
/* Active nav button (type=primary) */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
    background: #1C1C1C !important;
    color: #E5E5E5 !important;
    border-left: 3px solid #E50914 !important;
    border-radius: 0 6px 6px 0 !important;
    font-weight: 600 !important;
}

/* ── Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div { background:var(--dark3) !important; border-color:var(--border) !important; color:var(--text) !important; }
.stSelectbox label, .stSlider label, .stMultiSelect label { color: var(--muted) !important; }

/* ── Divider / scrollbar ── */
hr { border-color:var(--border) !important; margin:1.25rem 0 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--dark); }
::-webkit-scrollbar-thumb { background:var(--dark4); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────

def render_stars(rating: float, max_r: float = 10.0) -> str:
    filled = round((float(rating or 0) / max_r) * 5)
    return "&#9733;" * filled + "&#9734;" * (5 - filled)

def movie_card_html(movie: pd.Series, score=None, score_label="Match", show_pred=False) -> str:
    poster = str(movie.get("poster", ""))
    if poster and poster != "N/A" and poster.startswith("http"):
        poster_html = f'<img src="{poster}" alt="" loading="lazy" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;"/>' 
    else:
        poster_html = '<div style="position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:3rem;background:linear-gradient(135deg,#1a1a2e,#16213e);">&#127916;</div>'

    imdb = float(movie.get("imdb_rating", 0) or 0)
    score_b = f'<div class="score-badge">&#11088; {imdb}</div>' if imdb > 0 else ""
    match_b = ""
    if score is not None and float(score) > 0:
        pct = min(int(float(score) * 100), 99) if float(score) <= 1 else int(float(score))
        match_b = f'<div class="match-badge">{pct}% {score_label}</div>'

    year = movie.get("year", "")
    rt = int(movie.get("runtime", 0) or 0)
    rt_str = f" &middot; {rt}m" if rt > 0 else ""
    stars = render_stars(imdb)

    tags = [g.strip() for g in str(movie.get("genre", "")).split(",") if g.strip() and g.strip() != "N/A"][:2]
    genre_html = "".join(
        f'<span style="background:#252525;border:1px solid #2A2A2A;color:#8C8C8C;font-size:0.6rem;padding:2px 5px;border-radius:3px;white-space:nowrap;">{t}</span>'
        for t in tags
    )

    pred_html = ""
    if show_pred:
        pred = float(movie.get("predicted_rating", 0) or 0)
        if pred > 0:
            pred_html = f'<div style="font-size:0.7rem;color:#F5C518;margin-bottom:3px;">&#11014; Pred: {pred:.1f}/5</div>'

    title = str(movie.get("title", ""))
    title_safe = title.replace('"', '&quot;').replace("'", "&#39;")
    title_display = title[:21] + "&#8230;" if len(title) > 22 else title

    return (
        '<div class="movie-card">' +
        '<div style="position:relative;width:100%;padding-top:148%;overflow:hidden;background:#1C1C1C;">' +
        poster_html + score_b + match_b +
        '</div>' +
        '<div style="padding:0.7rem;">' +
        f'<div style="font-weight:600;font-size:0.83rem;color:#E5E5E5;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{title_safe}">{title_display}</div>' +
        f'<div style="font-size:0.7rem;color:#8C8C8C;margin-bottom:3px;">{year}{rt_str}</div>' +
        f'<div style="color:#F5C518;font-size:0.82rem;margin-bottom:3px;">{stars}</div>' +
        pred_html +
        f'<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:4px;">{genre_html}</div>' +
        '</div></div>'
    )

def movie_grid(df: pd.DataFrame, score_col=None, score_label="Match", show_pred=False, ncols=5):
    """Render entire grid as one st.markdown call to prevent Streamlit escaping inner HTML."""
    if df.empty:
        st.warning("No movies found.")
        return
    col_pct = int(100 / ncols)
    cards_html = ""
    for _, row in df.iterrows():
        sc = row[score_col] if score_col and score_col in row else None
        card = movie_card_html(row, sc, score_label, show_pred)
        cards_html += f'<div style="width:{col_pct}%;padding:0 6px 12px 6px;box-sizing:border-box;">{card}</div>'
    grid_html = f'<div style="display:flex;flex-wrap:wrap;margin:0 -6px;">{cards_html}</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  DATA & MODEL LOADING
# ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_everything():
    base = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(base, "data", "movies_cache.json")

    with st.spinner("🎬 Loading movie data from OMDB..."):
        raw_df = load_or_fetch_movies(cache)

    movies_df = preprocess_movies(raw_df)
    ratings_df = generate_synthetic_ratings(movies_df, n_users=50, seed=42)

    cb = ContentBasedRecommender()
    cb.fit(movies_df)

    cf = CollaborativeFilteringRecommender(n_factors=20)
    cf.fit(ratings_df, movies_df)

    return movies_df, ratings_df, cb, cf


# ──────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────

def sidebar(movies_df):
    """Sidebar with session_state-based navigation — works correctly on Streamlit Cloud."""
    # Initialise navigation state
    PAGES = ["🏠 Home", "🎯 Content-Based", "👥 Collaborative", "📊 Analytics"]
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGES[0]

    with st.sidebar:
        # ── Logo ──
        st.markdown("""
        <div style="padding:0.5rem 0 1.5rem;text-align:center;">
            <div class="logo-text">NeuralFlix</div>
            <div class="logo-sub">Neural · Intelligence · Cinema</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.68rem;letter-spacing:0.14em;color:#555;text-transform:uppercase;margin-bottom:0.5rem;padding:0 4px;">Navigate</div>', unsafe_allow_html=True)

        # ── Nav buttons (more reliable than st.radio on Streamlit Cloud) ──
        nav_items = [
            ("🏠", "Home"),
            ("🎯", "Content-Based"),
            ("👥", "Collaborative"),
            ("📊", "Analytics"),
        ]
        for icon, label in nav_items:
            full = f"{icon} {label}"
            is_active = st.session_state.current_page == full
            # Highlight active page with a red left border indicator
            indicator = "🔴" if is_active else "⬜"
            btn_label = f"{icon}  {label}"
            if st.button(
                btn_label,
                key=f"nav_{label}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state.current_page = full
                st.rerun()

        st.markdown("---")

        # ── Stats ──
        st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#555;text-transform:uppercase;margin-bottom:0.6rem;padding:0 4px;">Library Stats</div>', unsafe_allow_html=True)
        valid = movies_df[movies_df["imdb_rating"] > 0]
        avg_r = valid["imdb_rating"].mean() if not valid.empty else 0
        n_g = len(set(g.strip() for gg in movies_df["genre"]
                      for g in str(gg).split(",") if g.strip() and g.strip() != "N/A"))
        for val, lbl in [(len(movies_df), "Movies"), (f"{avg_r:.1f}★", "Avg Rating"), (n_g, "Genres"), (50, "Users")]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:5px 4px;border-bottom:1px solid #1C1C1C;">
                <span style="font-size:0.76rem;color:#555;">{lbl}</span>
                <span style="font-size:0.82rem;font-weight:600;color:#E5E5E5;">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.65rem;color:#444;text-align:center;line-height:1.7;">NeuralFlix v1.0<br>OMDB · TF-IDF · SVD<br>Built with Streamlit</div>', unsafe_allow_html=True)

    return st.session_state.current_page


# ──────────────────────────────────────────────────────
#  PAGE: HOME
# ──────────────────────────────────────────────────────

def pg_home(movies_df):
    st.markdown("""
    <div class="hero-banner">
        <div class="logo-text" style="font-size:2.6rem;">NeuralFlix</div>
        <div style="font-size:1rem;color:#aaa;margin-top:0.5rem;font-weight:300;max-width:580px;line-height:1.6;">
            Discover films you'll love with <strong style="color:#E50914;">AI-powered</strong>
            recommendations — Content Analysis and Collaborative Intelligence.
        </div>
    </div>""", unsafe_allow_html=True)

    valid = movies_df[movies_df["imdb_rating"] > 0]
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        (len(movies_df), "Movies in Library"),
        (f"{valid['imdb_rating'].mean():.1f}", "Avg IMDB Rating"),
        ("3", "AI Algorithms"),
        ("50", "User Profiles"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], stats):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Recommendation Engines</div>', unsafe_allow_html=True)
    a1,a2 = st.columns(2)
    algos = [
        ("🔍","Content-Based Filtering","Analyzes movie DNA — genre, director, cast and plot — to find films that share your movie's essence.","TF-IDF + Cosine Similarity"),
        ("👥","Collaborative Filtering","Learns from how similar users rated movies to predict your next watch — collective intelligence at work.","SVD Matrix Factorization"),
    ]
    for col, (icon, title, desc, tech) in zip([a1,a2], algos):
        with col:
            st.markdown(f"""<div class="algo-card">
                <div class="algo-icon">{icon}</div>
                <div class="algo-title">{title}</div>
                <div class="algo-desc">{desc}</div>
                <div class="algo-tech">{tech}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top Rated <span class="section-badge">IMDB</span></div>', unsafe_allow_html=True)
    movie_grid(valid.nlargest(10, "imdb_rating"), ncols=5)

    st.markdown('<div class="section-header">Modern Classics <span class="section-badge">2010+</span></div>', unsafe_allow_html=True)
    movie_grid(movies_df[movies_df["year"] >= 2010].nlargest(10, "imdb_rating"), ncols=5)


# ──────────────────────────────────────────────────────
#  PAGE: CONTENT-BASED
# ──────────────────────────────────────────────────────

def pg_content(movies_df, cb_model):
    st.markdown('<div class="section-header">🔍 Content-Based Filtering</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-panel">
        <div class="algo-title">How It Works</div>
        <div class="algo-desc">Each movie is converted into a <strong>TF-IDF vector</strong> from its genre, director, cast &amp; plot.
        We then compute <strong>Cosine Similarity</strong> between all pairs to identify the most content-similar films.
        Pick a movie you like and find its nearest neighbors in "movie-space".</div>
    </div>""", unsafe_allow_html=True)

    titles = sorted(movies_df["title"].tolist())
    c1, c2 = st.columns([4, 1])
    with c1:
        sel = st.selectbox("🎬 Select a movie you like:", titles)
    with c2:
        top_n = st.slider("Results", 5, 20, 10)

    # Show selected movie
    row = movies_df[movies_df["title"] == sel]
    if not row.empty:
        m = row.iloc[0]
        st.markdown("---")
        pc, ic = st.columns([1, 3])
        with pc:
            p = str(m.get("poster", ""))
            if p and p != "N/A" and p.startswith("http"):
                st.image(p, width=180)
            else:
                st.markdown('<div style="font-size:4rem;text-align:center;">🎬</div>', unsafe_allow_html=True)
        with ic:
            imdb = m.get("imdb_rating", 0) or 0
            st.markdown(f"""
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;letter-spacing:0.05em;">{m['title']} ({m.get('year','')})</div>
            <div style="color:#8C8C8C;font-size:0.8rem;margin-bottom:0.4rem;">{m.get('genre','')} · {m.get('director','')} · {m.get('runtime','')} min</div>
            <div style="color:#F5C518;font-size:0.88rem;margin-bottom:0.6rem;">⭐ {imdb}/10 &nbsp; {render_stars(float(imdb))}</div>
            <div style="font-size:0.81rem;color:#ccc;line-height:1.6;">{str(m.get('plot',''))[:300]}</div>""",
            unsafe_allow_html=True)
            actors = m.get("actors", "")
            if actors and actors != "N/A":
                st.markdown(f'<div style="font-size:0.76rem;color:#555;margin-top:0.5rem;">🎭 {actors}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-header">Similar to <em style="color:#E50914;">{sel}</em></div>', unsafe_allow_html=True)

    with st.spinner("Computing TF-IDF similarity..."):
        results = cb_model.get_similar_movies(sel, top_n=top_n)

    if results.empty:
        st.error("Movie not found. Try another title.")
    else:
        movie_grid(results, score_col="similarity_score", score_label="Match", ncols=5)
        with st.expander("📊 Full Similarity Table"):
            disp = results[["title","year","genre","director","imdb_rating","similarity_pct"]].copy()
            disp.columns = ["Title","Year","Genre","Director","IMDB","Match %"]
            disp["Match %"] = disp["Match %"].apply(lambda x: f"{x}%")
            st.dataframe(disp, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────
#  PAGE: COLLABORATIVE
# ──────────────────────────────────────────────────────

def pg_collab(movies_df, ratings_df, cf_model):
    st.markdown('<div class="section-header">👥 Collaborative Filtering</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-panel">
        <div class="algo-title">How It Works</div>
        <div class="algo-desc">A <strong>50 × N user-movie rating matrix</strong> is decomposed via
        <strong>SVD (Singular Value Decomposition)</strong> into latent factor matrices.
        Hidden factors capture taste patterns — "loves dark thrillers", "Nolan fan" — enabling
        rating predictions for movies a user hasn't seen yet.</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        uid = st.selectbox("👤 Select User ID:", sorted(ratings_df["user_id"].unique().tolist()), index=2)
    with c2:
        top_n = st.slider("Recommendations", 5, 20, 10)

    lf = cf_model.get_latent_factors_info()
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("SVD Factors (k)", lf.get("n_factors","N/A"))
    with m2: st.metric("Variance Explained", f"{lf.get('total_variance_explained',0)*100:.1f}%")
    with m3: st.metric(f"User {uid} Ratings", len(ratings_df[ratings_df["user_id"]==uid]))

    st.markdown("---")
    hc, rc = st.columns([1, 2])

    with hc:
        st.markdown(f'<div class="section-header" style="font-size:1.05rem;">User {uid} History</div>', unsafe_allow_html=True)
        hist = cf_model.get_user_history(uid, ratings_df, top_n=8)
        if not hist.empty:
            for _, r in hist.iterrows():
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.5rem;padding:4px 0;border-bottom:1px solid #1C1C1C;">
                    <div style="flex:1;font-size:0.78rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r.get('title','')}</div>
                    <div style="font-size:0.72rem;color:#F5C518;white-space:nowrap;">{"⭐"*int(round(r.get('rating',0)))} {r.get('rating',0):.1f}</div>
                </div>""", unsafe_allow_html=True)

    with rc:
        st.markdown(f'<div class="section-header" style="font-size:1.05rem;">Predicted for User {uid}</div>', unsafe_allow_html=True)
        with st.spinner("Running SVD matrix factorization..."):
            recs = cf_model.recommend_for_user(uid, top_n=top_n, only_unrated=True)
        movie_grid(recs, show_pred=True, ncols=4)

    with st.expander("🔬 SVD Latent Factor Visualization"):
        sv = lf.get("top_singular_values", [])
        evr = lf.get("explained_variance_ratio", [])
        if sv:
            fig = go.Figure(go.Bar(
                x=[f"Factor {i+1}" for i in range(len(sv))], y=evr,
                marker_color=["#E50914"] + ["#333"]*(len(evr)-1),
                text=[f"{v*100:.1f}%" for v in evr], textposition="outside",
            ))
            fig.update_layout(title="Variance Explained by SVD Factors",
                               paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
                               font_color="#E5E5E5", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────
#  PAGE: ANALYTICS
# ──────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
#  PAGE: ANALYTICS
# ──────────────────────────────────────────────────────

def pg_analytics(movies_df, ratings_df):
    st.markdown('<div class="section-header">📊 Library Analytics</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🎭 Genres", "⭐ Ratings", "📅 Timeline"])

    with tab1:
        gd = get_genre_distribution(movies_df)
        top_g = dict(list(gd.items())[:15])
        fig = px.bar(x=list(top_g.values()), y=list(top_g.keys()), orientation="h",
                     color=list(top_g.values()), color_continuous_scale=["#1C1C1C","#E50914"],
                     labels={"x":"Movies","y":"Genre"})
        fig.update_layout(title="Movies per Genre", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
                           font_color="#E5E5E5", height=430, showlegend=False,
                           coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        valid = movies_df[movies_df["imdb_rating"] > 0]["imdb_rating"]
        fig = px.histogram(valid, nbins=20, color_discrete_sequence=["#E50914"],
                            labels={"value":"IMDB Rating","count":"Movies"})
        fig.add_vline(x=valid.mean(), line_dash="dash", line_color="#F5C518",
                       annotation_text=f"Mean: {valid.mean():.2f}", annotation_font_color="#F5C518")
        fig.update_layout(title="IMDB Rating Distribution", paper_bgcolor="#1A1A1A",
                           plot_bgcolor="#1A1A1A", font_color="#E5E5E5", height=350,
                           showlegend=False, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

        sc_df = movies_df[(movies_df["imdb_rating"] > 0) & (movies_df["imdb_votes"] > 100)].copy()
        if not sc_df.empty:
            fig2 = px.scatter(sc_df, x="imdb_votes", y="imdb_rating",
                               hover_name="title", color="imdb_rating", size="imdb_votes",
                               color_continuous_scale=["#555","#E50914","#F5C518"],
                               labels={"imdb_votes":"Votes","imdb_rating":"Rating"}, size_max=28)
            fig2.update_layout(title="Rating vs Popularity", paper_bgcolor="#1A1A1A",
                                plot_bgcolor="#1A1A1A", font_color="#E5E5E5",
                                height=360, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        dd = get_decade_distribution(movies_df)
        fig = px.bar(x=list(dd.keys()), y=list(dd.values()),
                     color=list(dd.values()), color_continuous_scale=["#1C1C1C","#E50914"],
                     labels={"x":"Decade","y":"Movies"})
        fig.update_layout(title="Movies by Decade", paper_bgcolor="#1A1A1A",
                           plot_bgcolor="#1A1A1A", font_color="#E5E5E5", height=360,
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        yc = movies_df[movies_df["year"] >= 1990].groupby("year").size().reset_index()
        yc.columns = ["Year","Count"]
        fig2 = px.line(yc, x="Year", y="Count", color_discrete_sequence=["#E50914"], markers=True)
        fig2.update_layout(title="Movies per Year (1990+)", paper_bgcolor="#1A1A1A",
                            plot_bgcolor="#1A1A1A", font_color="#E5E5E5", height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────

def main():
    movies_df, ratings_df, cb, cf = load_everything()
    page = sidebar(movies_df)

    if "Home" in page:
        pg_home(movies_df)
    elif "Content-Based" in page:
        pg_content(movies_df, cb)
    elif "Collaborative" in page:
        pg_collab(movies_df, ratings_df, cf)
    elif "Analytics" in page:
        pg_analytics(movies_df, ratings_df)


if __name__ == "__main__":
    main()
