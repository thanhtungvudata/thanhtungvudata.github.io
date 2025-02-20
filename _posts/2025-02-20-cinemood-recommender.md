---
title: "🚀 From API to App: Creating a Mood-Based Trending Movie Recommender with Python, Hugging Face, and XGBoost"
date: 2025-02-20
categories:
  - Data Science Projects
tags:
  - Data Science
  - Multiclass Classification
  - ML Engineer
---
Ever wondered how to get personalized movie recommendations based on your mood? In this project, I built CineMood (a Mood-Based Trending Movie Recommendation Web App) from scratch—using Python, Hugging Face, XGBoost, and the TMDb API. The app analyzes trending movies, classifies them by mood, and delivers real-time recommendations.

In this post, I'll walk you through the entire process of a end-to-end machine learning project—from data collection and model training to deploying the app on Hugging Face Spaces.

<img src="/assets/images/cinemood_overview.png" alt="CineMood" width="600">


## **💡 Project Overview**
CineMood recommends movies based on six emotions:
- ❤️ Love – Romantic and heartwarming movies
- 😃 Joy – Feel-good, uplifting films
- 😲 Surprise – Unexpected twists and exciting plots
- 😢 Sadness – Emotional and tear-jerking stories
- 😨 Fear – Thrilling and chilling experiences
- 😡 Anger – Intense and dramatic narratives

The web app:
- Fetches trending movies from the TMDb API.
- Classifies each movie’s overview into one of the six moods using a XGBoost model.
- Caches results weekly to ensure fast recommendations.
- Suggests 3 unique trending movies per mood, refreshing automatically if fewer than 3 are found.

Key Features:
1. Mood-Based Recommendations:
    - Users select their mood from a dropdown.
    - The app recommends trending movies tailored to that mood.
2. Auto-Refresh Cache Weekly:
    - Movie classification runs once per week to keep the app fast and updated.
    - Recommendations update automatically as TMDb trends change.
3. Trending Movie Filtering:
    - Movies are sorted by release date (newest first).
    - Only movies released before the current week are considered.
4. Cloud Deployment:
    - The app is deployed on Hugging Face Spaces (free) for easy accessibility.

## **📂 Step 1: Organize Your Project**
Create a directory structure:
```plaintext
cinemood_project/
├── app.py                          # Streamlit web app
├── data/
│   ├── movie_mood_dataset.csv      # Generated dataset
├── models/
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorization of movie overview
│   ├── xgb_mood_classifier.pkl     # Trained XGBoost model
├── generate_movie_mood_dataset.py  # Model training script
├── train_model.py                  # Model training script
├── requirements.txt                # List of dependencies
├── README.md                       # Project documentation
```

## **📊 Step 2:  Data Collection**
- Fetched more than 1000 trending movies from the TMDb API, including weekly, top-rated, and popular movies.
- Built a custom movie mood dataset by classifying movie overviews with a pre-trained Hugging Face emotion classification model.
- Stored metadata like title, overview, poster, mood, and release date.
- Created a balanced (as much as possible) dataset of 200 movies per mood for training.

Create a file `generate_movie_mood_dataset.py` below to generate dataset `movie_mood_dataset.csv`:
```python
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from tqdm import tqdm

# Load API keys from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Hugging Face Emotion Classification Model
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# TMDb API Endpoints
TMDB_ENDPOINTS = [
    "https://api.themoviedb.org/3/trending/movie/week",
    "https://api.themoviedb.org/3/movie/top_rated",
    "https://api.themoviedb.org/3/movie/popular"
]

# Define target samples per mood
TARGET_SAMPLES_PER_MOOD = 200

# Dictionary to store movies per mood
movie_moods = {
    "joy": [], "sadness": [], "love": [], "anger": [], "fear": [], "surprise": []
}

# Set to track unique movie titles
unique_movie_titles = set()

def get_movies_from_tmdb(endpoint, page=1):
    """Fetch movies from TMDb API based on the given endpoint and page number."""
    try:
        response = requests.get(endpoint, params={"api_key": TMDB_API_KEY, "page": page}, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching movies from {endpoint}: {e}")
        return []

def classify_mood(movie_overview):
    """Classify movie mood using the Hugging Face emotion classifier."""
    if not movie_overview or len(movie_overview) < 10:
        return None
    try:
        result = classifier(movie_overview)
        mood = result[0]["label"]
        return mood if mood in movie_moods else None
    except Exception as e:
        print(f"❌ Error during mood classification: {e}")
        return None

def collect_movie_data():
    """Fetch movies, classify moods, and ensure 200 samples per mood."""
    for endpoint in TMDB_ENDPOINTS:
        print(f"📥 Fetching movies from {endpoint}...")
        page = 1

        while not all(len(movies) >= TARGET_SAMPLES_PER_MOOD for movies in movie_moods.values()):
            movies = get_movies_from_tmdb(endpoint, page)
            if not movies:
                break

            for movie in tqdm(movies, desc=f"Processing page {page}"):
                title, overview = movie.get("title"), movie.get("overview")
                if not title or not overview or title in unique_movie_titles:
                    continue

                mood = classify_mood(overview)
                if mood and len(movie_moods[mood]) < TARGET_SAMPLES_PER_MOOD:
                    movie_moods[mood].append({"Movie_Title": title, "Overview": overview, "Mood": mood})
                    unique_movie_titles.add(title)

            page += 1

            # Stop when each mood reaches its target
            if all(len(movies) >= TARGET_SAMPLES_PER_MOOD for movies in movie_moods.values()):
                break

def save_dataset():
    """Save the collected movie data into a CSV file."""
    all_movies = []
    for mood, movies in movie_moods.items():
        all_movies.extend(movies)

    df = pd.DataFrame(all_movies)
    df.to_csv("data/movie_mood_dataset.csv", index=False)
    print("✅ Movie mood dataset saved as movie_mood_dataset.csv")

if __name__ == "__main__":
    print("🚀 Collecting movies and ensuring 200 per mood...")
    collect_movie_data()
    save_dataset()
    print("🎬 Dataset generation complete!")
```

## **🤖 Step 3: Mood Classification**
- Leveraged an XGBoost Classifier trained on the movie dataset.
- Preprocessed text with TF-IDF vectorization and handled class imbalance natuarally by using XGBoost.
- Mapped Hugging Face labels to custom moods (e.g., “joy” → 😃 Joy).

Create a file `train_model.py`:
```python
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

print("🚀 Starting XGBoost model training with hyperparameter tuning ...")

# Load dataset
df = pd.read_csv("data/movie_mood_dataset.csv")
print(f"📂 Dataset loaded successfully. Total samples: {df.shape[0]}")

# Encode moods into numerical labels
label_encoder = LabelEncoder()
df["Mood_Label"] = label_encoder.fit_transform(df["Mood"])
print(f"🔢 Mood labels encoded. Unique moods: {len(label_encoder.classes_)}")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=2000,  # Limit features to avoid overfitting
    stop_words="english"
)
X = vectorizer.fit_transform(df["Overview"])
y = df["Mood_Label"]
print(f"📊 TF-IDF vectorization complete. Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📉 Data split into training ({X_train.shape[0]}) and test ({X_test.shape[0]}) sets.")


sample_weights = compute_sample_weight('balanced', y_train)

# Define base XGBoost classifier with class imbalance handling
base_xgb = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    random_state=42
)

# Define hyperparameter grid for tuning
param_grid = {
    'max_depth': [6, 8],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 500],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Perform GridSearchCV for hyperparameter tuning
print("🔍 Performing hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

# Best model after tuning
best_xgb_model = grid_search.best_estimator_
print(f"🏆 Best hyperparameters found: {grid_search.best_params_}")

# Calculate sample weights
sample_weights = compute_sample_weight("balanced", y_train)

# Train the best model on full training set
print("⏳ Training best model with optimized parameters...")
best_xgb_model.fit(X_train, y_train, verbose=True)
print("✅ Model training complete.")

# Save model and vectorizer
joblib.dump(best_xgb_model, "models/xgb_mood_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("💾 Model and vectorizer saved successfully.")

# Predictions on the test set
print("🔍 Generating predictions on test set...")
y_pred = best_xgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 XGBoost Model Accuracy: {accuracy:.2%}")

# Classification report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("🚀 Training pipeline completed successfully!")
```

## **🤖 Step 4: Web App with Streamlit**
- Built an interactive Streamlit app for users to select their mood.
- Displayed 3 unique movie recommendations per mood with posters and descriptions.
- Cached movie classifications for one week to improve load time.
- If fewer than 3 recommendations were found, fetched additional pages from the TMDb API.

Create a file `app.py`:
```python
import streamlit as st
import pandas as pd
import joblib
import requests
import random
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta

# Load environment variables (API keys)
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Check if key is available
if not TMDB_API_KEY:
    raise ValueError("🚨 TMDB_API_KEY not found! Please set it in Hugging Face Secrets.")

# Cache settings
CACHE_FILE = "movies_cache.pkl"
CACHE_EXPIRATION_DAYS = 7  # Refresh once per week

# Load trained model and vectorizer
model = joblib.load("models/xgb_mood_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Define mood labels and corresponding emotion icons in the desired order
mood_mapping = {
    "love": ("love", "❤️"),
    "joy": ("joy", "😃"),
    "surprise": ("surprise", "😲"),
    "sadness": ("sadness", "😢"),
    "fear": ("fear", "😨"),
    "anger": ("anger", "😡"),
}

# Hugging Face original order to custom order mapping
huggingface_to_custom = {
    "anger": "anger",
    "fear": "fear",
    "joy": "joy",
    "love": "love",
    "sadness": "sadness",
    "surprise": "surprise"
}

# TMDb API endpoint and image URL
WEEK_ENDPOINT = "https://api.themoviedb.org/3/trending/movie/week"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# Get the first day of the current week (Monday)
first_day_of_current_week = datetime.now() - timedelta(days=datetime.now().weekday())
current_week = datetime.now().isocalendar()[1]  # ISO week number

# 🕰 Cache movie fetching for one week
@st.cache_data(ttl=60 * 60 * 24 * 7, hash_funcs={int: str})
def fetch_trending_movies(week=current_week):
    """Fetch trending movies from TMDb and classify them once per week."""
    movies_cache = []
    page = 1

    while len(movies_cache) < 150:  # Fetch enough movies for all moods
        try:
            response = requests.get(WEEK_ENDPOINT, params={"api_key": TMDB_API_KEY, "page": page})
            response.raise_for_status()
            results = response.json().get("results", [])

            for movie in results:
                title = movie.get("title")
                overview = movie.get("overview")
                poster = TMDB_IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None
                release_date = movie.get("release_date")

                if title and overview and release_date:
                    release_date_obj = datetime.strptime(release_date, "%Y-%m-%d")
                    if release_date_obj < first_day_of_current_week:  # Ensure the movie was released before this week
                        hf_mood = classify_mood(overview)
                        custom_mood = huggingface_to_custom.get(hf_mood, "unknown")
                        movies_cache.append({
                            "title": title,
                            "overview": overview,
                            "poster": poster,
                            "mood": custom_mood,
                            "release_date": release_date
                        })

            page += 1
            if not results:
                break
        except Exception as e:
            st.error(f"Failed to fetch trending movies (Page {page}): {e}")
            break

    # Sort by release date (newest first)
    movies_cache.sort(key=lambda x: x["release_date"], reverse=True)
    return movies_cache

def classify_mood(movie_overview):
    """Predict movie mood using XGBoost model and map to custom order."""
    X = vectorizer.transform([movie_overview])
    mood_label = model.predict(X)[0]
    hf_mood = ["anger", "fear", "joy", "love", "sadness", "surprise"][mood_label]
    return hf_mood

def fetch_recommendations(user_mood):
    """Fetch 3 recommendations per mood from cached trending movies. Get more if fewer than 3."""
    mood_movies = []
    page = 1

    while len(mood_movies) < 3:
        trending_movies = fetch_trending_movies(current_week)

        # Filter movies by user mood
        for movie in trending_movies:
            if movie["mood"] == user_mood and movie["title"] not in [m["title"] for m in mood_movies]:
                mood_movies.append(movie)
                if len(mood_movies) >= 3:
                    break

        # If fewer than 3, fetch more pages
        if len(mood_movies) < 3:
            try:
                response = requests.get(WEEK_ENDPOINT, params={"api_key": TMDB_API_KEY, "page": page})
                response.raise_for_status()
                results = response.json().get("results", [])

                for movie in results:
                    title = movie.get("title")
                    overview = movie.get("overview")
                    poster = TMDB_IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None
                    release_date = movie.get("release_date")

                    if title and overview and release_date:
                        release_date_obj = datetime.strptime(release_date, "%Y-%m-%d")
                        if release_date_obj < first_day_of_current_week:
                            hf_mood = classify_mood(overview)
                            custom_mood = huggingface_to_custom.get(hf_mood, "unknown")
                            if custom_mood == user_mood and title not in [m["title"] for m in mood_movies]:
                                mood_movies.append({
                                    "title": title,
                                    "overview": overview,
                                    "poster": poster,
                                    "mood": custom_mood,
                                    "release_date": release_date
                                })

                page += 1
                if not results:
                    break
            except Exception as e:
                st.error(f"Failed to fetch additional trending movies: {e}")
                break

    return mood_movies[:3]

# Streamlit UI
st.title("🎬 CineMood: Get Your Mood-Based Trending Movies! ⚡")

# User selects their mood
user_mood, mood_icon = st.selectbox(
    "Select your mood:",
    [(mood, emoji) for mood, (mood, emoji) in mood_mapping.items()],
    format_func=lambda x: f"{x[1]} {x[0]}"
)

# Fetch recommendations based on user mood
recommended_movies = fetch_recommendations(user_mood)

# Display recommendations
st.subheader(f"{mood_icon} Recommended Trending Movies for Your Mood: {user_mood.capitalize()}")

if recommended_movies:
    for movie in recommended_movies:
        st.markdown(f"### 🎬 {movie['title']} ({movie['release_date']})")
        st.write(f"📖 {movie['overview']}")
        if movie['poster']:
            st.image(movie['poster'], width=200)
        st.write("---")
else:
    st.write("❌ No matching movies found. Try again later!")

# Footer Section
st.markdown("**Made by [Thanh Tung Vu](https://thanhtungvudata.github.io/)**")
```

## **🤖 Step 5: Deployment**
- Deployed the app to Hugging Face Spaces using Streamlit.
- Free hosting with weekly cache refresh for up-to-date recommendations.

Deployment steps:
- Create `requirements.txt`
- Push code to GitHub
- Deploy to Hugging Face Spaces

The file `requirements.txt`:
```plaintext
requests
pandas
numpy
scikit-learn
xgboost
joblib
transformers
huggingface_hub
fastapi
uvicorn
tmdbv3api
python-dotenv
tqdm
streamlit
```


## **📦Tech Stack**
- Backend: Python, XGBoost, TMDb API, bert-base-uncased-emotion (Pre-trained Hugging Face Emotional Classification Model)
- Frontend: Streamlit
- Deployment: Docker, Hugging Face Spaces
- Data Processing: Pandas, NumPy, Scikit-Learn

## **🎉 Results and Live Demo**

The final web app delivers mood-based movie recommendations in just a second, with fresh content every week. 

You can try it here:👉 [CineMood Live App on Hugging Face Spaces](https://huggingface.co/spaces/thanhtungvudata/cinemood)

## 🚀 Conclusion
CineMood showcases how machine learning, APIs, and web tools can create an engaging and user-friendly app. From data collection to deployment, it demonstrates the power of end-to-end ML pipelines.

Looking ahead, next steps will be:
- 🔍 Expand mood categories for more nuanced recommendations.
- 💡 Improve classification accuracy with BERT embeddings.
- 📈 Integrate user feedback to refine suggestions.

The code of this project is available [here](https://github.com/thanhtungvudata/mood_based_trending_movie_recommendation). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
