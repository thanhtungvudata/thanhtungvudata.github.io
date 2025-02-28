---
title: "AI-Agent-Assisted Mood-Based Movie Recommendation"
date: 2025-02-27
image: /assets/images/cinemood2_overview.png
categories:
  - Data Science Projects
tags:
  - Data Science
  - AI Agent
  - ML Engineer
---
[CineMood](https://medium.com/@tungvu_37498/from-api-to-app-creating-a-mood-based-trending-movie-recommender-with-python-hugging-face-model-e32d67b492e2) started as a **machine-learning-based** mood-based movie recommendation system that analyzed user input and recommended trending movies from **The Movie Database (TMDB)**. However, it had limitations in dynamically matching moods with real-time trending movies.  
With the rise of **AI agents**, we upgraded **CineMood to CineMood2** in this post, making it an **LLM-powered** system that leverages **GPT-4o-mini** to enhance movie recommendations.  

<img src="/assets/images/cinemood2_overview.png" alt="CineMood" width="600">

---

## 🎯 Project Overview  

### **🎬 CineMood → CineMood2: What Changed?**  
1. **CineMood (ML-Based Approach)**  
   - Used **predefined mood-to-genre mapping**.
   - Limited adaptability to **dynamic** trending movies.

2. **CineMood2 (AI-Agent-Based Approach)**  
   - Uses **GPT-4o-mini** to **extract** moods from user input.  
   - **Fetches 100 trending movies** from TMDB.
   - Uses **GPT-4o-mini** again to find the **3 most relevant** movies based on mood and movie overviews.
   - More **dynamic** trending movies.

---

## ❌ Why ChatGPT Alone Cannot Recommend Trending Movies  

While **ChatGPT** is a powerful language model, it lacks access to **real-time** trending movies.  

- **ChatGPT cannot fetch fresh data** from TMDB on its own.
- **It relies on training data**, which may be outdated.  
- **It does not have access to external APIs** like TMDB.

However, **GPT-4o-mini can still be used as a tool** to analyze moods and **rank movies** once we fetch the latest trending data from TMDB.

---

## 🤖 Why GPT-4o-Mini?  

It is my personal choice. Any model (e.g., Mistral-7B, Llama2, Claude, etc.) can be used for this project. 

---

## 🛠️ Technical Steps in CineMood2  

### **1️⃣ Extracting Mood from User Input**  
- When a user enters how they feel, **GPT-4o-mini** extracts **3 mood words**.  
- Example:  
  - **User Input:** "I feel relaxed and peaceful today."  
  - **Extracted Moods:** `["calm", "peaceful", "serene"]`  

### **2️⃣ Fetching Trending Movies from TMDB**  
- **Get 100 trending movies** for the week.  
- **Filter out movies** that have **not been released yet**.  
- Movies are **ranked from the latest to oldest** by release date.

### **3️⃣ Matching Movies to User’s Mood**  
- **GPT-4o-mini** analyzes the **movie overviews** and finds the **3 best matches**.  
- It also provides a **match explanation** for each movie.  

  - **Example Output:**  
    ```
    🎬 Movie: "Forrest Gump"
    📅 Release Date: 1994-07-06
    ✅ Match Reason: This movie embodies warmth, nostalgia, and optimism, matching the user's mood perfectly.
    ```

---

## 🚀 How CineMood2 is Built (Using Docker & Streamlit)  

### 🔹 **Step 1: Setting Up Streamlit UI (`app.py`)**
💡 Goal: Provide a simple user interface where users describe how they feel.

✅ How it Works:

1. The Streamlit UI presents a text box where users type their current mood.
2. A button click triggers the recommendation process.
3. User input is processed and passed to GPT-4o-mini for mood detection.

🛠 Technical Details:
1. The app listens for user input via st.text_area().
2. When the user clicks "Find Movies," the input is passed to detect_mood() from `llm.py`.
3. The detected mood words are displayed in the UI.

Create `app.py`:
```python
import streamlit as st

from llm import detect_mood, get_movies_by_mood
from tmdb_api import fetch_movies

st.set_page_config(
    page_title="🎬 Mood-Based Trending Movie Recommendation", layout="centered"
)

st.title("🎬 CineMood2: Get your Mood-based Trending Movies!⚡")

user_mood = st.text_area("💬 How do you feel right now?", st.session_state.get("user_mood", ""), height=100)

if st.button("Find Movies"):
    if user_mood.strip():
        with st.spinner("🔍 Analyzing your mood..."):
            mood_words = detect_mood(user_mood)
        st.success(f"🤖 AI Detected Moods: {', '.join(mood_words).title()}")

        with st.spinner("🎥 Fetching movies and ranking matches..."):
            movies = fetch_movies(60)
            recommended_movies = get_movies_by_mood(mood_words, movies)

        if recommended_movies:
            for movie in recommended_movies:
                st.subheader(movie["title"])
                st.write(f"📅 Release Date: {movie['release_date']}")
                st.write(f"🎭 Match Reason: {movie['match_reason']}")
                if movie["poster"]:
                    st.image(movie["poster"], width=200)
                st.write(f"📜 Overview: {movie['overview']}")
                st.markdown("---")
        else:
            st.warning("⚠️ No suitable movie recommendations found.")
    else:
        st.warning("⚠️ Please enter how you feel to get movie recommendations.")

# Footer Section
st.markdown("**Made by [Thanh Tung Vu](https://thanhtungvudata.github.io/)**")

```

### 🔹 **Step 2: Using OpenAI's GPT-4o-mini for Mood Detection (`llm.py`)**
💡 Goal: Extract key mood words from user input.

✅ How it Works:
1. The user’s input is sent to GPT-4o-mini via an API call.
2. The model returns three mood-related words that best describe the user’s feelings.

🛠 Technical Details:
1. detect_mood(user_input) sends the input to GPT-4o-mini via the OpenAI API.
2. The model is prompted to return exactly three words in a structured format.
3. If an error occurs, a default neutral mood is returned.

Create `llm.py`:
```python
import json

import openai

from config import OPENAI_API_KEY

# Initialize OpenAI API client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def detect_mood(user_input):
    """
    Uses GPT-4o-mini to detect mood from user input.
    Returns exactly 3 descriptive mood words.
    """
    prompt = f"""
    Analyze the following user input and determine the three best words to describe the mood.

    User input: "{user_input}"
    
    Respond with exactly 3 words, separated by commas.
    Example: happy, joyful, excited.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}]
        )
        mood_words = response.choices[0].message.content.strip().lower().split(", ")
        return mood_words if len(mood_words) == 3 else ["neutral", "neutral", "neutral"]
    except Exception as e:
        print(f"⚠️ Error with OpenAI API in detect_mood: {e}")
        return ["neutral", "neutral", "neutral"]


def get_movies_by_mood(mood_words, movies):
    """
    Uses GPT-4o-mini to rank movies based on how well their overview matches the detected mood words.
    Returns the top 3 movies with match explanations, sorted by release date (latest first).
    """
    if not movies:
        print("⚠️ No movies available to match moods.")
        return []

    movie_descriptions = "\n".join(
        [f"{i+1}. {m['title']}: {m['overview']}" for i, m in enumerate(movies)]
    )

    prompt = f"""
    You must output only valid JSON and nothing else.
    The JSON should be an array of exactly 3 objects.
    Each object must have two keys: "index" (an integer) and "match_reason" (a non-empty string).
    
    The user is in a mood described by these words: {", ".join(mood_words)}.
    
    Below are movie descriptions:
    {movie_descriptions}
    
    Select the top 3 movies that best match this mood and provide a brief explanation (1-2 sentences) for each.
    Respond strictly in JSON format:
    [
        {{"index": 1, "match_reason": "Explanation for movie 1"}},
        {{"index": 2, "match_reason": "Explanation for movie 2"}},
        {{"index": 3, "match_reason": "Explanation for movie 3"}}
    ]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}]
        )
        json_response = response.choices[0].message.content.strip()
        ranked_movies = json.loads(json_response)
        matched_movies = []
        default_explanation = "This movie appears to be a good match based on its emotional tone and themes."
        for entry in ranked_movies:
            index = entry.get("index", 0) - 1
            explanation = entry.get("match_reason", "").strip() or default_explanation
            if 0 <= index < len(movies):
                matched_movie = movies[index]
                matched_movie["match_reason"] = explanation
                matched_movies.append(matched_movie)
        # Ensure exactly 3 movies are returned by filling with fallbacks if necessary
        while len(matched_movies) < 3 and len(movies) >= 3:
            fallback = movies[len(matched_movies)]
            fallback["match_reason"] = default_explanation
            matched_movies.append(fallback)
        # Sort the matched movies by release date (latest first)
        matched_movies = sorted(
            matched_movies, key=lambda x: x["release_date"], reverse=True
        )
        return matched_movies[:3]
    except Exception as e:
        print(f"⚠️ Error ranking movies: {e}")
        fallback_movies = []
        default_explanation = "This movie appears to be a good match based on its emotional tone and themes."
        for m in movies[:3]:
            m["match_reason"] = default_explanation
            fallback_movies.append(m)
        fallback_movies = sorted(
            fallback_movies, key=lambda x: x["release_date"], reverse=True
        )
        return fallback_movies

```

### 🔹 **Step 3: Fetching TMDB Trending Movies (`tmdb_api.py`)**
💡 Goal: Get a list of currently trending movies from The Movie Database (TMDB) API.

✅ How it Works:
1. The app fetches 100 trending movies using TMDB's API.
2. Movies without overviews or future releases are filtered out.
3. Movies are sorted by release date (newest first).

🛠 Technical Details:
1. fetch_movies(max_movies=100) retrieves movies using TMDB’s /trending/movie/week endpoint.
2. Each movie entry is checked for:
- A valid release date (must be in the past).
- A non-empty movie overview.
3. The results are sorted by release date, with the most recent movies first.

Create `tmdb_api.py`:
```python
import datetime

import requests

from config import TMDB_API_KEY


def get_first_day_of_week():
    """Returns the first day (Monday) of the current week."""
    today = datetime.date.today()
    return today - datetime.timedelta(days=today.weekday())


def fetch_movies(max_movies=100):
    """
    Fetch up to `max_movies` trending movies, ensuring only movies with release dates before the first day
    of the current week are considered, and that they have non-empty overviews.
    Returns the movies sorted by release date (latest first).
    """
    movies = []
    pages_to_fetch = (max_movies // 20) + 1
    first_day_of_week = get_first_day_of_week()

    for page in range(1, pages_to_fetch + 1):
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}&language=en-US&page={page}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for movie in data.get("results", []):
                release_date = movie.get("release_date", "9999-12-31")
                overview = movie.get("overview", "").strip()
                try:
                    release_date_obj = datetime.datetime.strptime(
                        release_date, "%Y-%m-%d"
                    ).date()
                except ValueError:
                    continue
                if overview and release_date_obj < first_day_of_week:
                    movies.append(
                        {
                            "title": movie["title"],
                            "overview": overview,
                            "poster": (
                                f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                if movie.get("poster_path")
                                else None
                            ),
                            "release_date": release_date,
                        }
                    )
            if len(movies) >= max_movies:
                break
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching movies: {e}")
            break
    return sorted(movies, key=lambda x: x["release_date"], reverse=True)[:max_movies]

```

### 🔹 **Step 4: Using GPT-4o-mini to Find Best Matches (llm.py)**
💡 Goal: Select the top 3 movies that best match the user's mood.

✅ How it Works:
1. The 100 trending movies are sent to GPT-4o-mini.
2. The model compares movie overviews against the 3 detected mood words.
3. Top 3 movies are selected based on relevance, each with a brief explanation.

🛠 Technical Details:
1. get_movies_by_mood(mood_words, movies) processes the movies using GPT-4o-mini.
2. The model is prompted to return structured JSON, ensuring:
- Three movie indices are selected.
- Each movie has a match explanation.
3. If errors occur, the first three movies are used with a default explanation.

### 🔹 **Step 5: Deploying CineMood2 in Docker**
💡 Goal: Ensure consistent deployment and easy sharing across environments.

✅ How it Works:
1. The entire project is containerized using Docker.
2. A Dockerfile defines:
- Base image (Python + dependencies).
- Installation of Streamlit, OpenAI API, and TMDB API tools.
- Instructions to run the Streamlit app inside the container.
3. The app can now be deployed anywhere, ensuring reproducibility.

🛠 Technical Details:

- Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

- Create `docker-compose.yml`:

```yml
version: '3.8'

services:
  mood-movie-app:
    build: .
    container_name: mood-movie-app
    ports:
      - "8501:8501"
    restart: always
```

- Docker Commands:
    - Build: `docker-compose up --build`
    - Re-Run: `docker-compose down`, then `docker-compose up --build`

## **📦Tech Stack**
- AI & Backend: GPT-4o-mini, TMDB API, Python 
- Frontend: Streamlit
- Development & Deployment: Docker, VSCode, Flake8, pytest 

## **🎉 Results and Live Demo**

The final web app delivers mood-based movie recommendations in just a second, with fresh content every week. 

You can try it here:👉 [CineMood Live App on Hugging Face Spaces](https://huggingface.co/spaces/thanhtungvudata/cinemood2)

<div style="text-align: center;">
    <h3>🎬 Try CineMood Now!</h3>
    <iframe
        src="https://thanhtungvudata-cinemoodv2.hf.space"
        width="100%"
        height="600"
        style="border: none; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);"
        allowfullscreen>
    </iframe>
</div>

## 📌 Conclusion
CineMood2 provides an AI-powered, mood-based movie recommendation system that seamlessly integrates GPT-4o-mini, TMDB trending movies, and Docker deployment.

💡 Key Advantages: 
- Uses real-time trending data instead of pre-trained ChatGPT knowledge.
- Optimized for cost (GPT-4o-mini) and performance (Cloud API instead of local LLM).
- Fully containerized for easy deployment and scalability.

🚀 Next Steps:
- Improve user feedback collection for better recommendations.
- Add multi-language support for global users.

The code of this project is available [here](https://github.com/thanhtungvudata/mood_based_trending_movie_recommendation). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
