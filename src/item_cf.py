import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("Loading matrix...")

user_item_matrix = pd.read_csv(
    "data/user_item_matrix.csv",
    index_col=0
)

print("Calculating item similarity...")

item_similarity = cosine_similarity(
    user_item_matrix.T
)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print(item_similarity_df.shape)