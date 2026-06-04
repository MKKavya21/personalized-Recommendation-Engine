import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load matrix
user_item_matrix = pd.read_csv(
    "data/user_item_matrix.csv",
    index_col=0
)

# Similarity
user_similarity = cosine_similarity(
    user_item_matrix
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

target_user = 1

# Most similar users
similar_users = user_similarity_df[target_user] \
    .sort_values(ascending=False)

print("Top Similar Users:")
print(similar_users.head(10))