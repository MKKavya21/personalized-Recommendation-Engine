import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Recommendation Engine",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Cold Start Recommendations",
        "Content-Based Recommendations",
        "Deep Learning",
        "Hybrid Recommendation",
        "Traditional vs Deep Learning",
        "Model Architecture"
    ]
)

# ==================================================
# PROJECT OVERVIEW
# ==================================================

if page == "Project Overview":

    st.title("🎬 Personalized Recommendation Engine")

    st.write(
        """
        Recommendation System POC using MovieLens Dataset.

        Implemented:
        - User-Based Collaborative Filtering
        - Item-Based Collaborative Filtering
        - SVD Matrix Factorization
        - ALS Matrix Factorization
        - Content-Based Filtering
        - Deep Learning Recommender
        - Two-Tower Architecture
        """
    )

    st.header("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("RMSE", "0.8799")
    col2.metric("MAE", "0.6759")
    col3.metric("Precision", "0.8098")
    col4.metric("Recall", "0.3314")

    st.subheader("Evaluation Metrics Chart")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "RMSE",
                "MAE",
                "Precision",
                "Recall"
            ],
            "Value": [
                0.8799,
                0.6759,
                0.8098,
                0.3314
            ]
        }
    )

    st.bar_chart(
        metrics_df.set_index("Metric")
    )

# ==================================================
# COLD START RECOMMENDATIONS
# ==================================================

elif page == "Cold Start Recommendations":

    st.title("Cold Start Recommendations")

    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    movie_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )

    popular_movies = movie_stats[
        movie_stats["rating_count"] > 50
    ]

    popular_movies = popular_movies.sort_values(
        by="avg_rating",
        ascending=False
    )

    recommendations = popular_movies.merge(
        movies,
        on="movieId"
    )

    st.dataframe(
        recommendations[
            ["title", "avg_rating", "rating_count"]
        ].head(10)
    )

# ==================================================
# CONTENT-BASED RECOMMENDATIONS
# ==================================================

elif page == "Content-Based Recommendations":

    st.title("Content-Based Recommendations")

    movies = pd.read_csv("data/movies.csv")

    movies["genres"] = movies["genres"].fillna("")

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        movies["genres"]
    )

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    indices = pd.Series(
        movies.index,
        index=movies["title"]
    ).drop_duplicates()

    selected_movie = st.selectbox(
        "Choose a Movie",
        movies["title"]
    )

    if st.button("Recommend Similar Movies"):

        idx = indices[selected_movie]

        sim_scores = list(
            enumerate(cosine_sim[idx])
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        sim_scores = sim_scores[1:11]

        movie_indices = [
            i[0]
            for i in sim_scores
        ]

        recommendations = movies[
            "title"
        ].iloc[movie_indices]

        st.subheader("Recommended Movies")

        for movie in recommendations:
            st.write(movie)

# ==================================================
# DEEP LEARNING
# ==================================================

elif page == "Deep Learning":

    st.title("Deep Learning Recommender")

    st.write(
        """
        Neural Recommendation Model using
        User Embeddings and Movie Embeddings
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Epoch 1 Loss", "260.25")
    col2.metric("Epoch 5 Loss", "78.23")
    col3.metric("Predicted Rating", "4.73")

    st.subheader("Architecture")

    st.markdown(
        """
        User Embedding (50D)

        +

        Movie Embedding (50D)

        ↓

        Neural Network

        ↓

        Predicted Rating
        """
    )

# ==================================================
# HYBRID RECOMMENDATION
# ==================================================

elif page == "Hybrid Recommendation":

    st.title("Hybrid Recommendation Engine")

    st.write(
        """
        Multiple recommendation signals are
        combined to generate a final score.
        """
    )

    hybrid_df = pd.DataFrame(
        {
            "Component": [
                "SVD Score",
                "Content Score",
                "Popularity Score"
            ],
            "Weight": [
                0.4,
                0.3,
                0.3
            ]
        }
    )

    st.subheader("Blending Layer")

    st.dataframe(hybrid_df)

    st.subheader("Final Formula")

    st.code(
        """
Final Score =
0.4 × SVD Score
+
0.3 × Content Score
+
0.3 × Popularity Score
        """
    )
    # ==================================================
# TRADITIONAL VS DEEP LEARNING
# ==================================================

elif page == "Traditional vs Deep Learning":

    st.title("Traditional vs Deep Learning Comparison")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "User-Based CF",
                "Item-Based CF",
                "SVD",
                "ALS",
                "Content-Based",
                "Deep Learning",
                "Two-Tower"
            ],
            "Category": [
                "Traditional",
                "Traditional",
                "Traditional",
                "Traditional",
                "Traditional",
                "Deep Learning",
                "Deep Learning"
            ]
        }
    )

    st.subheader("Implemented Models")

    st.dataframe(comparison_df)

    st.subheader("Traditional Recommendation Methods")

    st.write("""
    Traditional recommendation methods use similarity calculations
    and matrix factorization techniques.

    Models Implemented:
    - User-Based Collaborative Filtering
    - Item-Based Collaborative Filtering
    - SVD Matrix Factorization
    - ALS Matrix Factorization
    - Content-Based Filtering

    Advantages:
    - Easy to explain
    - Faster training
    - Lower computational requirements
    """)

    st.subheader("Deep Learning Recommendation Methods")

    st.write("""
    Deep learning approaches learn embeddings for users and movies.

    Models Implemented:
    - Neural Recommendation Model
    - Two-Tower Architecture

    Advantages:
    - Learns complex user-item relationships
    - Captures non-linear interactions
    - Better scalability
    """)

    analysis_df = pd.DataFrame(
        {
            "Aspect": [
                "Interpretability",
                "Training Speed",
                "Computational Cost",
                "Scalability",
                "Complex Pattern Learning"
            ],
            "Traditional Methods": [
                "High",
                "Fast",
                "Low",
                "Good",
                "Limited"
            ],
            "Deep Learning Methods": [
                "Moderate",
                "Slower",
                "Higher",
                "Excellent",
                "Strong"
            ]
        }
    )

    st.subheader("Comparative Analysis")

    st.dataframe(analysis_df)

    st.success(
        """
        Traditional recommendation methods provide strong baseline performance
        and are easier to explain.

        Deep learning models capture richer user-item interactions
        and scale better for large recommendation systems.

        The Hybrid Recommendation Engine combines both approaches
        to improve recommendation quality.
        """
    )

# ==================================================
# MODEL ARCHITECTURE
# ==================================================

elif page == "Model Architecture":

    st.title("Model Architecture")

    architecture = pd.DataFrame(
        {
            "Model": [
                "User CF",
                "Item CF",
                "SVD",
                "ALS",
                "Content-Based",
                "Deep Learning",
                "Two-Tower"
            ],
            "Status": [
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed"
            ]
        }
    )

    st.dataframe(architecture)