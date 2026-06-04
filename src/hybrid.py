import pandas as pd

def hybrid_score(
    svd_score,
    content_score,
    popularity_score
):

    # Normalize SVD from 0-5 → 0-1
    svd_score = svd_score / 5

    final_score = (
        0.4 * svd_score +
        0.3 * content_score +
        0.3 * popularity_score
    )

    return final_score


svd = 4.5
content = 0.85
popularity = 0.90

score = hybrid_score(
    svd,
    content,
    popularity
)

print("Hybrid Score:", round(score, 4))