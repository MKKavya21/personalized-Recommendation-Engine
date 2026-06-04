# 🎬 Personalized Recommendation Engine

A complete end-to-end Movie Recommendation System built using Collaborative Filtering, Matrix Factorization, Content-Based Filtering, Deep Learning, and Hybrid Recommendation techniques.

---

# 🚀 Live Demo

Streamlit Application:

https://personalized-recommendation-engine-jdltsnrvckqyxzq5rfxmni.streamlit.app/

---

# 💻 Source Code

GitHub Repository:

https://github.com/MKKavya21/personalized-Recommendation-Engine

---

# 📌 Project Overview

This project demonstrates multiple recommendation approaches used in real-world recommendation systems such as Netflix, Amazon, Spotify, and YouTube.

The system combines:

* Collaborative Filtering
* Matrix Factorization
* Content-Based Filtering
* Deep Learning Recommendation
* Two-Tower Neural Architecture
* Hybrid Recommendation Engine

The goal is to recommend relevant movies to users while improving recommendation accuracy and personalization.

---

# 📊 Results Summary

| Metric      | Score  |
| ----------- | ------ |
| RMSE        | 0.8826 |
| MAE         | 0.6780 |
| Precision@K | 0.8067 |
| Recall@K    | 0.3326 |

---

# 🧠 Models Implemented

## Traditional Recommendation Models

### User-Based Collaborative Filtering

Finds users with similar rating behavior and recommends movies liked by similar users.

### Item-Based Collaborative Filtering

Finds movies that are frequently liked together and recommends similar items.

### SVD Matrix Factorization

Learns latent user and movie factors for rating prediction.

### ALS Matrix Factorization

Alternating Least Squares based recommendation model.

### Content-Based Filtering

Uses movie genres and user-provided tags to recommend similar movies.

---

## Deep Learning Models

### Deep Learning Recommender

Implemented using PyTorch.

Architecture:

User Embedding
+
Movie Embedding
↓
Dense Neural Network
↓
Predicted Rating

Training Results:

* Initial Loss: 1.7859
* Final Loss: 1.0802
* Predicted Rating Example: 3.33

---

### Two-Tower Architecture

Separate embedding towers are used for users and movies.

Architecture:

User Tower
↓
User Embedding

Movie Tower
↓
Movie Embedding

Dot Product Similarity
↓
Recommendation Score

Training Results:

* Epoch 1 Loss: 44.8368
* Epoch 2 Loss: 44.6864
* Epoch 3 Loss: 44.5365

---

# 🔥 Hybrid Recommendation Engine

The final recommendation score combines multiple recommendation signals.

Formula:

Final Score =
0.4 × SVD Score +
0.3 × Content Score +
0.3 × Popularity Score

This improves recommendation quality by leveraging multiple models simultaneously.

---

# ❄️ Cold Start Recommendation

For new users with no interaction history:

* Popular movies are identified
* Average ratings are calculated
* Frequently rated movies are prioritized

This solves the cold-start problem for new users.

---

# 📈 Dashboard Features

The Streamlit dashboard includes:

* Project Overview
* Model Performance Metrics
* Cold Start Recommendations
* Content-Based Recommendations
* Deep Learning Results
* Hybrid Recommendation Engine
* Traditional vs Deep Learning Comparison
* Model Architecture Overview

---

# 📂 Project Structure

personalized-Recommendation-Engine/

├── dashboard.py

├── README.md

├── requirements.txt

├── data/

│   ├── movies.csv

│   ├── ratings.csv

│   ├── tags.csv

│   └── links.csv

└── src/

```
├── user_cf.py

├── item_cf.py

├── svd_model.py

├── als_model.py

├── content_based.py

├── hybrid.py

├── deep_learning_recommender.py

└── two_tower_model.py
```

---

# ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/MKKavya21/personalized-Recommendation-Engine.git
```

Navigate to Project Folder

```bash
cd personalized-Recommendation-Engine
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Dashboard

```bash
python -m streamlit run dashboard.py
```

---

# 🖼️ Dashboard Screenshots

## Project Overview

![Project Overview](screenshots/overview.png)

## Cold Start Recommendations

![Cold Start](screenshots/cold_start.png)

## Content-Based Recommendations

![Content Based](screenshots/content_based.png)

## Deep Learning Recommender

![Deep Learning](screenshots/deep_learning.png)

## Hybrid Recommendation Engine

![Hybrid Recommendation](screenshots/hybrid.png)

## Traditional vs Deep Learning

![Comparison](screenshots/comparison.png)

## Model Architecture

![Architecture](screenshots/architecture.png)

---

# 👩‍💻 Author

Kavya


