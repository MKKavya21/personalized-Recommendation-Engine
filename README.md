# 🎬 Personalized Recommendation Engine

A complete end-to-end Movie Recommendation System built using Collaborative Filtering, Matrix Factorization, Content-Based Filtering, Deep Learning, and Hybrid Recommendation techniques.

## 🚀 Live Demo

Streamlit Application:

https://personalized-recommendation-engine-jdltsnrvckqyxzq5rfxmni.streamlit.app/

## 💻 Source Code

GitHub Repository:

https://github.com/MKKavya21/personalized-Recommendation-Engine

---

## 📊 Results Summary

| Metric      | Score  |
| ----------- | ------ |
| RMSE        | 0.8826 |
| MAE         | 0.6780 |
| Precision@K | 0.8067 |
| Recall@K    | 0.3326 |

---

## ✨ Key Features

* User-Based Collaborative Filtering
* Item-Based Collaborative Filtering
* SVD Matrix Factorization
* ALS Matrix Factorization
* Content-Based Filtering
* Deep Learning Recommender
* Two-Tower Architecture
* Cold Start Recommendation
* Hybrid Recommendation Engine
* Streamlit Interactive Dashboard
* Cloud Deployment using Streamlit Community Cloud

---

## 📂 Project Structure

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

├── user_cf.py

├── item_cf.py

├── svd_model.py

├── als_model.py

├── content_based.py

├── hybrid.py

├── deep_learning_recommender.py

└── two_tower_model.py

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/MKKavya21/personalized-Recommendation-Engine.git
```

Navigate to project folder:

```bash
cd personalized-Recommendation-Engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit Dashboard:

```bash
python -m streamlit run dashboard.py
```
