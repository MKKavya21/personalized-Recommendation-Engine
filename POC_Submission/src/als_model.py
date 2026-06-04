from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

print("Starting Spark Session...")

spark = SparkSession.builder \
    .appName("MovieLens_ALS") \
    .getOrCreate()

print("Loading ratings...")

ratings = spark.read.csv(
    "data/ratings.csv",
    header=True,
    inferSchema=True
)

ratings = ratings.select(
    "userId",
    "movieId",
    "rating"
)

print("Training ALS model...")

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    rank=10,
    maxIter=10,
    regParam=0.1
)

model = als.fit(ratings)

print("ALS Model Trained Successfully!")

spark.stop()