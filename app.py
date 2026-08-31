import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0b0f19;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Movie card */
    .movie-card {
        background: #151b29;
        border: 1px solid #263044;
        border-radius: 16px;
        padding: 15px;
        min-height: 430px;
        margin-bottom: 15px;
    }

    .movie-title {
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 10px;
    }

    .movie-meta {
        color: #aeb7c7;
        font-size: 0.9rem;
        margin-top: 5px;
    }

    .rating {
        color: #fbbf24;
        font-weight: 700;
    }

    /* Section heading */
    .section-title {
        color: white;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Info boxes */
    .info-box {
        background: #111827;
        border: 1px solid #263044;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }

    .info-number {
        font-size: 1.7rem;
        font-weight: 700;
        color: white;
    }

    .info-label {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f1420;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL FILES
# =========================================================

@st.cache_resource
def load_model():

    with open("df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return df, indices, tfidf_matrix


df, indices, tfidf_matrix = load_model()


# =========================================================
# OMDB API
# =========================================================

API_KEY = st.secrets["OMDB_API_KEY"]


@st.cache_data(ttl=3600)
def get_movie_details(movie_name):

    url = "https://www.omdbapi.com/"

    params = {
        "apikey": API_KEY,
        "t": movie_name
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if data.get("Response") == "True":
            return data

    except requests.exceptions.RequestException:
        return None

    return None


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend(movie_name, n=5):

    if movie_name not in indices:
        return []

    idx = indices[movie_name]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:n+1]

    recommendations = []

    for movie_idx in similar_indices:

        recommendations.append({
            "title": df["title"].iloc[movie_idx],
            "similarity": similarity_scores[movie_idx]
        })

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎬 CineMatch")

    st.caption(
        "Machine Learning Movie Recommendation System"
    )

    st.divider()

    st.markdown("### ⚙️ Recommendation Settings")

    number_of_movies = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** TF-IDF + Cosine Similarity")

    st.write("**Recommendation:** Content-Based")

    st.write("**Movie Metadata:** OMDb API")

    st.divider()

    st.markdown("### 💡 How it works")

    st.caption(
        "The system compares movie metadata using "
        "TF-IDF vectors and cosine similarity to find "
        "movies with similar characteristics."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover your next favorite movie using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# STATS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-number">{len(df):,}</div>
            <div class="info-label">Movies in Database</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-box">
            <div class="info-number">TF-IDF</div>
            <div class="info-label">Feature Extraction</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="info-box">
            <div class="info-number">Cosine</div>
            <div class="info-label">Similarity Algorithm</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MOVIE SELECTION
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Find Similar Movies</div>',
    unsafe_allow_html=True
)

movie_list = (
    df["title"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

selected_movie = st.selectbox(
    "Select a movie you like",
    movie_list,
    index=movie_list.index("Toy Story")
    if "Toy Story" in movie_list else 0
)


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns([1, 5])

with col1:

    recommend_button = st.button(
        "🚀 Recommend",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🔄 Clear",
        use_container_width=False
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

if recommend_button:

    recommendations = recommend(
        selected_movie,
        number_of_movies
    )

    if not recommendations:

        st.error(
            "Sorry, recommendations could not be generated."
        )

    else:

        st.markdown(
            f'<div class="section-title">'
            f'✨ Movies similar to "{selected_movie}"'
            f'</div>',
            unsafe_allow_html=True
        )

        # Selected movie information

        selected_details = get_movie_details(selected_movie)

        if selected_details:

            with st.expander(
                f"🎥 About {selected_movie}",
                expanded=False
            ):

                about_col1, about_col2 = st.columns([1, 3])

                with about_col1:

                    poster = selected_details.get("Poster")

                    if poster and poster != "N/A":
                        st.image(
                            poster,
                            use_container_width=True
                        )

                with about_col2:

                    st.markdown(
                        f"### {selected_details.get('Title', selected_movie)}"
                    )

                    st.write(
                        f"⭐ IMDb Rating: "
                        f"{selected_details.get('imdbRating', 'N/A')}"
                    )

                    st.write(
                        f"📅 Year: "
                        f"{selected_details.get('Year', 'N/A')}"
                    )

                    st.write(
                        f"🎭 Genre: "
                        f"{selected_details.get('Genre', 'N/A')}"
                    )

                    st.write(
                        selected_details.get(
                            "Plot",
                            "Plot unavailable."
                        )
                    )

        st.markdown(
            '<div class="section-title">'
            '🎞️ Recommended For You'
            '</div>',
            unsafe_allow_html=True
        )

        # Create movie columns

        cols = st.columns(number_of_movies)

        for col, recommendation in zip(
            cols,
            recommendations
        ):

            movie_name = recommendation["title"]

            similarity = recommendation["similarity"]

            details = get_movie_details(movie_name)

            with col:

                if details:

                    poster = details.get("Poster")

                    if poster and poster != "N/A":

                        st.image(
                            poster,
                            use_container_width=True
                        )

                    st.markdown(
                        f'<div class="movie-title">'
                        f'{details.get("Title", movie_name)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    rating = details.get(
                        "imdbRating",
                        "N/A"
                    )

                    year = details.get(
                        "Year",
                        "N/A"
                    )

                    genre = details.get(
                        "Genre",
                        "N/A"
                    )

                    st.markdown(
                        f'<div class="movie-meta">'
                        f'⭐ <span class="rating">{rating}</span>'
                        f' &nbsp; | &nbsp; 📅 {year}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(genre)

                    st.progress(
                        min(float(similarity), 1.0)
                    )

                    st.caption(
                        f"Similarity: {similarity:.1%}"
                    )

                    with st.expander("📖 Plot"):

                        st.write(
                            details.get(
                                "Plot",
                                "Plot unavailable."
                            )
                        )

                else:

                    st.warning(
                        f"Details unavailable for {movie_name}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:

    st.caption(
        "🎬 CineMatch • Content-Based Recommendation System"
    )

with footer_col2:

    st.caption(
        "Built with Python • Scikit-learn • Streamlit • OMDb API"
    )