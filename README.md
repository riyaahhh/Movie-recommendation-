# 🎬 Movie Recommendation System

A **Content-Based Movie Recommendation System** built using **Python and Scikit-learn** that suggests movies similar to a given movie based on metadata such as genres, keywords, cast, and movie overview.

The system processes **5000+ movies** from the TMDB dataset and generates the **top 5 most similar movies** using **Natural Language Processing (NLP)** and **Cosine Similarity**.

---

## 📌 Project Overview

Movie recommendation systems are widely used by streaming platforms to suggest relevant content to users.
This project demonstrates a **content-based filtering approach**, where recommendations are made based on the similarity between movie features.

Each movie is represented as a **feature vector** generated from textual metadata. Similarity between movies is computed using **cosine similarity**, and the most similar movies are returned as recommendations.

---

## 🚀 Features

* Content-based movie recommendation
* Uses movie metadata such as:

  * Genres
  * Keywords
  * Cast
  * Director
  * Movie overview
* NLP text processing and feature engineering
* Vectorization using **CountVectorizer**
* Similarity computation using **Cosine Similarity**
* Generates **Top 5 movie recommendations**

---

## 🧠 How the System Works

Pipeline:

Movie Dataset
↓
Feature Extraction (genres + keywords + cast + overview + director)
↓
Text Processing
↓
Vectorization using CountVectorizer
↓
Cosine Similarity Calculation
↓
Top 5 Recommended Movies

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP (CountVectorizer)

Python Libraries:

* pandas
* numpy
* sklearn
* ast

---

## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset** available on **Kaggle**.

Dataset Files:

tmdb_5000_movies.csv
tmdb_5000_credits.csv

These files contain movie metadata including cast, crew, genres, keywords, and plot overview.

---

## 📂 Project Structure

movie-recommender
│
├── main.py
├── movies.csv
├── credits.csv
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1. Clone the Repository

git clone https://github.com/yourusername/movie-recommender.git

### 2. Navigate to Project Folder

cd movie-recommender

### 3. Install Dependencies

pip install pandas numpy scikit-learn

Or use requirements file:

pip install -r requirements.txt

---

## ▶️ Running the Project

Run the Python script:

python main.py

Enter a movie name when prompted.

Example:

Enter a movie name: Avatar

Output:

🎬 Recommended Movies for: Avatar

1. Titan A.E.
2. Small Soldiers
3. Independence Day
4. Ender's Game
5. Aliens vs Predator: Requiem

---

## 📌 Example

Input:

Avatar

Output:

Titan A.E.
Independence Day
Ender's Game
Small Soldiers
Aliens vs Predator: Requiem

---

## ⭐ Future Improvements

Possible improvements for this project:

* Add **movie posters using TMDB API**
* Build a **web interface using Streamlit**
* Deploy the recommender system online
* Improve recommendations using **TF-IDF Vectorization**
* Implement **collaborative filtering**

---

## 👩‍💻 Author

**Riya Ratnani**

First-Year Engineering Student
AI & Data Science

Interested in:

* Machine Learning
* Mobile App Development
* Competitive Programming

GitHub: https://github.com
LinkedIn: https://linkedin.com

---

## 📜 License

This project is open-source and available under the MIT License.

