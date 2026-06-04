import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD

print("Loading ratings...")

ratings = pd.read_csv(
    "data/ratings.csv"
)

reader = Reader(
    rating_scale=(0.5, 5)
)

data = Dataset.load_from_df(
    ratings[
        ["userId", "movieId", "rating"]
    ],
    reader
)

print("Building training set...")

trainset = data.build_full_trainset()

print("Training SVD model...")

model = SVD()

model.fit(trainset)

print("Model trained successfully!")

prediction = model.predict(
    uid=1,
    iid=50
)

print("\nPrediction:")

print(
    f"User 1 predicted rating for Movie 50: {prediction.est:.2f}"
)