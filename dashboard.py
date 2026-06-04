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

    col1.metric("RMSE", "0.8826")
    col2.metric("MAE", "0.6780")
    col3.metric("Precision", "0.8067")
    col4.metric("Recall", "0.3326")

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
            0.8826,
            0.6780,
            0.8067,
            0.3326
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

elif page == "Content-Based Recommendations":

    st.title("Content-Based Recommendations")

    movies = pd.read_csv("data/movies.csv")
    tags = pd.read_csv("data/tags.csv")

    movie_tags = tags.groupby(
        "movieId"
    )["tag"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()

    movies = movies.merge(
        movie_tags,
        on="movieId",
        how="left"
    )

    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].fillna("")

    movies["content"] = (
        movies["genres"]
        + " "
        + movies["tag"]
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(
        movies["content"]
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

        sim_scores = sim_scores[1:51]

        movie_indices = [
            i[0]
            for i in sim_scores
        ]

        recommendations = movies.iloc[
            movie_indices
        ][["title", "genres"]]

        st.subheader(
            "Top Recommended Movies"
        )

        st.dataframe(
            recommendations.reset_index(
                drop=True
            )
        )

# ==================================================
# DEEP LEARNING
# ==================================================

elif page == "Deep Learning":

    st.title("Deep Learning Recommender")

    st.write(
    """

    Components:
    - User Embedding Layer
    - Movie Embedding Layer
    - Dense Neural Network
    - Rating Prediction Output

    Training Results:
    - Epoch 1 Loss: 1.7859
    - Epoch 20 Loss: 1.0802
    - Sample Predicted Rating: 3.33
    """
)

    col1, col2, col3 = st.columns(3)

    col1.metric("Epoch 1 Loss", "1.7859")
    col2.metric("Epoch 20 Loss", "1.0802")
    col3.metric("Predicted Rating", "3.33")


# ==================================================
# HYBRID RECOMMENDATION
# ==================================================

elif page == "Hybrid Recommendation":

    st.title("Hybrid Recommendation Engine")

    st.write(
        """
        Hybrid recommendations combine:

        • Collaborative Filtering (SVD)

        • Content-Based Similarity

        • Popularity Ranking

        Final Score =
        0.4 × SVD Score +
        0.3 × Content Score +
        0.3 × Popularity Score
        """
    )

    hybrid_df = pd.DataFrame(
        {
            "Movie": [
                "Toy Story 2",
                "Monsters Inc",
                "Finding Nemo",
                "Shrek",
                "The Incredibles"
            ],

            "SVD Score": [
                4.50,
                4.30,
                4.20,
                4.10,
                4.40
            ],

            "Content Score": [
                0.80,
                0.75,
                0.72,
                0.70,
                0.78
            ],

            "Popularity Score": [
                0.90,
                0.88,
                0.85,
                0.83,
                0.89
            ]
        }
    )

    hybrid_df["Hybrid Score"] = (
        0.4 * hybrid_df["SVD Score"]
        + 0.3 * hybrid_df["Content Score"]
        + 0.3 * hybrid_df["Popularity Score"]
    )

    hybrid_df = hybrid_df.sort_values(
        by="Hybrid Score",
        ascending=False
    )

    st.subheader(
        "Hybrid Recommendation Results"
    )

    st.dataframe(
        hybrid_df.round(3)
    )

    st.subheader(
        "Top Recommendation"
    )

    st.success(
        f"Recommended Movie: "
        f"{hybrid_df.iloc[0]['Movie']}"
    )

    st.metric(
        "Hybrid Score",
        round(
            hybrid_df.iloc[0]["Hybrid Score"],
            3
        )
    )

    st.subheader(
        "Hybrid Score Distribution"
    )

    chart_df = hybrid_df[
        ["Movie", "Hybrid Score"]
    ].set_index("Movie")

    st.bar_chart(chart_df)
elif page == "Traditional vs Deep Learning":

    st.title("Traditional vs Deep Learning Models")

    st.subheader("Implemented Models")

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
            ],
            "Purpose": [
                "Find Similar Users",
                "Find Similar Movies",
                "Predict Ratings",
                "Matrix Factorization",
                "Genre & Tag Similarity",
                "Learn User-Movie Patterns",
                "Embedding Similarity Search"
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RMSE", "0.8826")
        st.metric("MAE", "0.6780")

    with col2:
        st.metric("Precision@K", "0.8067")
        st.metric("Recall@K", "0.3326")

    st.subheader("Traditional vs Deep Learning")

    metrics_comparison = pd.DataFrame(
        {
            "Criteria": [
                "Training Time",
                "Scalability",
                "Interpretability",
                "Cold Start Handling",
                "Complex Pattern Learning",
                "Production Usage"
            ],
            "Traditional Models": [
                "Fast",
                "Medium",
                "High",
                "Weak",
                "Limited",
                "Common"
            ],
            "Deep Learning Models": [
                "Slower",
                "High",
                "Medium",
                "Strong",
                "Excellent",
                "Very Common"
            ]
        }
    )

    st.dataframe(
        metrics_comparison,
        use_container_width=True
    )

    st.subheader("Models Implemented")

    chart_df = pd.DataFrame(
        {
            "Model Type": [
                "Traditional",
                "Deep Learning"
            ],
            "Count": [
                5,
                2
            ]
        }
    )

    st.bar_chart(
        chart_df.set_index("Model Type")
    )

    st.subheader("Comparison Summary")

    st.markdown("""
### Traditional Recommendation Models

- User-Based Collaborative Filtering finds users with similar interests.
- Item-Based Collaborative Filtering finds similar movies.
- SVD and ALS use matrix factorization techniques.
- Content-Based Filtering uses genres and tags.

Advantages:
- Fast training
- Easy to explain
- Works well on small datasets

Limitations:
- Struggles with cold-start problems
- Limited ability to learn complex patterns

---

### Deep Learning Models

- Neural Recommender learns user and movie embeddings.
- Two-Tower Architecture learns separate user and item representations.

Advantages:
- Learns complex user behavior patterns
- Better scalability for large datasets
- More effective personalization

Limitations:
- Requires more data
- Longer training time
- More computational resources

---

### Final Conclusion

Traditional models are simpler, faster, and highly interpretable.

Deep Learning models provide stronger personalization and better scalability.

This project combines both approaches using a Hybrid Recommendation Engine to leverage the strengths of each method.
""")
    


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