import pandas as pd

# Load ratings dataset
ratings = pd.read_csv("data/ratings.csv")

# Show first 5 rows
print(ratings.head())

# Show column names
print("\nColumns:")
print(ratings.columns)

# Count users
print("\nNumber of Users:")
print(ratings["userId"].nunique())

# Count movies
print("\nNumber of Movies:")
print(ratings["movieId"].nunique())

# Count ratings
print("\nNumber of Ratings:")
print(len(ratings))

# Create User-Item Matrix
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Replace missing values with 0
user_item_matrix = user_item_matrix.fillna(0)

print("\nUser-Item Matrix Shape:")
print(user_item_matrix.shape)

# Save matrix
user_item_matrix.to_csv("data/user_item_matrix.csv")

print("\nUser Item Matrix Saved Successfully!")