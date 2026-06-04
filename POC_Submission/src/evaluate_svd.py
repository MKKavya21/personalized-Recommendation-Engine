import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD

from surprise.model_selection import train_test_split
from surprise import accuracy

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

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

print("Training SVD...")

model = SVD()

model.fit(trainset)

print("Making predictions...")

predictions = model.test(testset)

print("\nRMSE:")
accuracy.rmse(predictions)

print("\nMAE:")
accuracy.mae(predictions)