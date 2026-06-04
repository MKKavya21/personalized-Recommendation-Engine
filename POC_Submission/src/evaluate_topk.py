import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split

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

model = SVD(
    n_factors=100,
    n_epochs=50,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

model.fit(trainset)

predictions = model.test(testset)

THRESHOLD = 3.5

tp = 0
fp = 0
fn = 0

for pred in predictions:

    actual = pred.r_ui
    estimated = pred.est

    if estimated >= THRESHOLD:

        if actual >= THRESHOLD:
            tp += 1
        else:
            fp += 1

    elif actual >= THRESHOLD:
        fn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")