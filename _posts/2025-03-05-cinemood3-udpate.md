---
title: "Building a RAG Mood-Based Trending Movie Recommendation App"
date: 2025-03-05
image: /assets/images/cinemood2_overview.png
categories:
  - Data Science Projects
tags:
  - Data Science
  - AI Agent
  - ML Engineer
  - RAG
---
In my previous [post](https://thanhtungvudata.github.io/data%20science%20projects/cinemood2-issues/), I built a **mood-based movie recommendation app** that used **LLM (Large Language Model)** to analyze metadata from approximately **50 trending movies** and select the **top 3 movies** based on user mood. The app handled **validation and hallucination** by having the LLM compare user input against a predefined list of valid moods. However, the method faced challenges in working with **larger datasets**, ensuring **reliable validation**, and **reducing LLM hallucination.**

<img src="/assets/images/cinemood3_overview.png" alt="CineMood3" width="600">

In this post, I take a step further by implementing a **RAG (Retrieval-Augmented Generation) mood-based trending movie recommendation app** that can efficiently **handle a larger dataset of movies**, validate moods more effectively using **embeddings and similarity scores**, and improve the **explainability of recommendations**.

## **Limitations of the Previous Approach**
While the previous approach provided **decent recommendations**, it had several shortcomings:

1. **Scalability Issues**: LLM-based approaches require processing a predefined, **limited set of movies** (e.g., 50 movies) at inference time. Increasing the dataset size exponentially increases computation time and memory usage, leading to impractical delays.
2. **Validation of Mood**: The mood validation was **not robust**—it relied only on simple keyword matching, which fails to capture semantic similarity. This means they might misinterpret user moods, leading to **incorrect recommendations**.  
3. **LLM Hallucination**: Since the model relied solely on **LLM reasoning** to select movies, it could **hallucinate recommendations** not present in the dataset. This is because LLMs, when not grounded in structured data, **tend to hallucinate**—generating movie titles, summaries, or recommendations that do not exist because their responses are based purely on learned probabilities rather than factual data.


To overcome these limitations, I implemented **RAG (Retrieval-Augmented Generation)**, which enhances the accuracy and explainability of recommendations.

## **What is RAG?**
**RAG** is a method that combines **information retrieval (IR) and generative AI** to improve text generation by grounding responses in **real-world data**. Instead of relying **solely on the LLM’s internal knowledge**, RAG retrieves **relevant documents** or data points from an external **vector database** before generating responses.

## **RAG-Based Workflow in My App**

### Overview Diagram:

```bash
 ┌──────────────────────────┐
 │        User Input        │
 │ (Mood description in UI) │
 └──────────▲───────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │      OpenAI Embeddings   │  ◀── Generates vector embeddings for moods
 └──────────▲───────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │      ChromaDB (Moods)    │  ◀── Stores & retrieves valid mood embeddings
 └──────────▲───────────────┘
            │
     [Cosine Similarity]
            ▼
 ┌──────────────────────────┐
 │    Detected Closest Mood │
 └──────────▲───────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │      ChromaDB (Movies)   │  ◀── Stores & retrieves movie embeddings
 └──────────▲───────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │  Top 3 Movie Matches     │
 │ (Based on mood similarity) │
 └──────────▲───────────────┘
            │
     [GPT-4o-mini LLM]
            ▼
 ┌──────────────────────────┐
 │  AI-Generated Explanation│
 │   (Why this movie?)      │
 └──────────▲───────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │   Streamlit UI (Web App) │
 │ Displays Movie Titles,   │
 │ Posters & Explanations   │
 └──────────────────────────┘
```


### **Step 1: Storing Valid Mood and Movie Metadata in ChromaDB**
- **Valid moods** and **movie metadata** are **embedded using OpenAI embeddings** and stored in **ChromaDB**, a vector database optimized for retrieval.
- This enables **efficient vector-based retrieval** of moods and movies.

### **Step 2: User Mood Validation Using Embedding Similarity**
- The **user input mood** is embedded using **OpenAI embeddings**.
- The system retrieves **the top 5 closest moods** from ChromaDB.
- If the **best similarity score** is **below a defined threshold**, the system **rejects the input** and asks the user to rephrase their mood.

### **Step 3: Retrieving the Top 3 Movies Using Semantic Search**
- If the user mood is valid, the system **performs a semantic search in ChromaDB**.
- Movies are ranked based on **similarity between the top mood and the movie title, overview, and tagline**.
- The **top 3 movies** with the highest scores are selected.

### **Step 4: LLM Generates Explanation for Recommendations**
- The **LLM (GPT-4o-mini)** is prompted to explain why the **selected movies** match the user's mood.
- The model uses **retrieved metadata** to ensure the explanation is **grounded in real data**, reducing hallucination.

### **Step 5: Displaying the Recommendations in a Streamlit Web App**
- The **user sees the top 3 movies**, along with details:
  - 🎬 **Title**
  - 📅 **Release Date**
  - 🏷️ **Tagline**
  - 🎭 **Cast & Director**
  - 🌍 **Production Country**
  - 🏢 **Production Company**
  - ⏳ **Runtime**
  - 🖼️ **Movie Poster** (Fetched dynamically via URL)
  - 📝 **LLM-generated explanation** of why these movies fit the user's mood

## **Why is RAG a Better Method for This App?**
✅ **Scalable Retrieval**: Instead of embedding and processing all movies in a single LLM prompt, **RAG retrieves only the most relevant movie embeddings** from a **vector database** (e.g., ChromaDB, FAISS, Pinecone) at runtime. This enables the app to **efficiently scale to thousands or millions of movies** while maintaining fast response times.

✅ **Better Mood Validation**: By **embedding user inputs** and comparing them against a predefined set of **valid moods** using **cosine similarity in a high-dimensional vector space**, the system ensures that only moods with **high semantic relevance** are considered valid. This prevents misclassification and improves mood-based filtering.

✅ **Reduced LLM Hallucination**: Instead of relying on the LLM to generate recommendations from its internal knowledge, **RAG retrieves actual metadata from an external database**. The LLM only processes **verified movie details** (e.g., titles, descriptions, genres), ensuring that all recommendations are **based on real, retrievable content**.

✅ **Improved Ranking**: Standard LLM-based ranking often relies on simple text similarity measures (e.g., token-level similarity or TF-IDF-based heuristics), which do not effectively capture **latent relationships between moods and movie metadata**. Now, using **embedding-based similarity scoring**, the system computes the **cosine similarity between the user's mood embedding and movie metadata embeddings** (title, overview, tagline). This enables **context-aware ranking**, ensuring that the most semantically relevant movies are prioritized over those with only superficial text similarity.

## **Technical Details**
Below is a detailed step-by-step breakdown of the project along with the relevant code snippets.

### 1. **Fetching Trending Movies from TMDB**

#### Create `fetch_movies.py`: 

✅ Fetch trending movies from TMDB API.

✅ Retrieve detailed metadata (genres, cast, director, etc.).

✅ Filter out movies released after the current week.

✅ Save the processed dataset for embedding generation.

```python
import requests
import json
import time
from datetime import datetime, timedelta
from config import TMDB_API_KEY

# TMDB API Endpoints
TRENDING_URL = "https://api.themoviedb.org/3/trending/movie/week"
MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie/{movie_id}?api_key=" + TMDB_API_KEY + "&append_to_response=keywords"
CREDITS_URL = "https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key=" + TMDB_API_KEY

# Calculate the first day of the current week
current_date = datetime.utcnow()
first_day_of_week = current_date - timedelta(days=current_date.weekday())
first_day_of_week_str = first_day_of_week.strftime("%Y-%m-%d")

def fetch_trending_movies(pages=10):
    """
    Fetch trending movies from TMDB API over a given number of pages.

    Args:
        pages (int): Number of pages to fetch. Each page contains ~20 movies.

    Returns:
        list: A list of trending movies with basic metadata.
    """
    movies = []
    for page in range(1, pages + 1):
        response = requests.get(TRENDING_URL, params={"api_key": TMDB_API_KEY, "page": page})
        if response.status_code == 200:
            movies.extend(response.json().get("results", []))
        else:
            print(f"Error fetching movies from page {page}: {response.status_code}")
    return movies

def fetch_movie_details(movie_id):
    """
    Fetch detailed metadata for a movie, including cast, crew, and keywords.

    Args:
        movie_id (int): The ID of the movie to fetch details for.

    Returns:
        dict or None: A dictionary containing movie details if successful, otherwise None.
    """
    details_response = requests.get(MOVIE_DETAILS_URL.format(movie_id=movie_id))
    credits_response = requests.get(CREDITS_URL.format(movie_id=movie_id))
    
    if details_response.status_code == 200 and credits_response.status_code == 200:
        details = details_response.json()
        credits = credits_response.json()
        release_date = details.get("release_date")
        
        if release_date and release_date < first_day_of_week_str:
            return {
                "id": details.get("id"),
                "title": details.get("title"),
                "overview": details.get("overview"),
                "release_date": release_date,
                "popularity": details.get("popularity"),
                "vote_average": details.get("vote_average"),
                "vote_count": details.get("vote_count"),
                "genres": [genre["name"] for genre in details.get("genres", [])],
                "runtime": details.get("runtime"),
                "original_language": details.get("original_language"),
                "spoken_languages": [lang["english_name"] for lang in details.get("spoken_languages", [])],
                "status": details.get("status"),
                "budget": details.get("budget"),
                "revenue": details.get("revenue"),
                "production_companies": [company["name"] for company in details.get("production_companies", [])],
                "production_countries": [country["name"] for country in details.get("production_countries", [])],
                "tagline": details.get("tagline"),
                "homepage": details.get("homepage"),
                "imdb_id": details.get("imdb_id"),
                "poster_path": f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}" if details.get("poster_path") else None,
                "main_cast": [cast["name"] for cast in credits.get("cast", [])[:5]],
                "director": next((crew["name"] for crew in credits.get("crew", []) if crew["job"] == "Director"), "Unknown"),
                "keywords": [keyword["name"] for keyword in details.get("keywords", {}).get("keywords", [])],
            }
    else:
        print(f"Error fetching details for movie ID {movie_id}")
    return None

def get_trending_movies_with_details():
    """
    Fetch trending movies and retrieve their detailed metadata.

    Filters out movies released after the first day of the current week.

    Returns:
        list: A list of dictionaries containing detailed movie metadata.
    """
    trending_movies = fetch_trending_movies(pages=10)
    movies_metadata = []
    total_movies = len(trending_movies[:200])
    print(f"🔄 Fetching details for {total_movies} movies...")
    
    for index, movie in enumerate(trending_movies[:200], start=1):
        movie_id = movie["id"]
        movie_details = fetch_movie_details(movie_id)
        
        if movie_details:
            movies_metadata.append(movie_details)
        
        if index % 10 == 0 or index == total_movies:
            print(f"✅ Processed {index}/{total_movies} movies...")
    
    return movies_metadata

if __name__ == "__main__":
    """
    Main execution block:
    - Fetches trending movies
    - Retrieves detailed metadata
    - Saves the data to a JSON file
    """
    trending_movies_data = get_trending_movies_with_details()

    # Save data to a JSON file for use in embedding generation
    with open("trending_movies.json", "w") as f:
        json.dump(trending_movies_data, f, indent=4)

    print("\n✅ Trending movies saved to `trending_movies.json`")
```

#### Create `generate_embeddings.py`:

✅ Loads movie metadata from `trending_movies.json`.

✅ Uses OpenAI's `text-embedding-ada-002` model (via LangChain) to convert text data into embeddings.

✅ Stores the embeddings along with metadata in `movie_embeddings.json` for later use in ChromaDB.

```python
import json
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

# Load TMDB movie data from the previous step
with open("trending_movies.json", "r") as f:
    movies = json.load(f)

# Initialize LangChain OpenAI Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def generate_movie_embedding(movie):
    """
    Generate vector embeddings for a movie using OpenAI's embedding model.

    Args:
        movie (dict): A dictionary containing metadata about a movie.

    Returns:
        list: A vector representation of the movie's metadata in high-dimensional space.
    """
    text_data = f"""
    Title: {movie.get('title', 'Unknown')}
    Overview: {movie.get('overview', 'No overview available')}
    Genres: {', '.join(movie.get('genres', []))}
    Main Cast: {', '.join(movie.get('main_cast', []))}
    Director: {movie.get('director', 'Unknown')}
    Tagline: {movie.get('tagline', 'No tagline available')}
    Production Countries: {', '.join(movie.get('production_countries', []))}
    Keywords: {', '.join(movie.get('keywords', []))}
    Runtime: {movie.get('runtime', 'Unknown')} minutes
    Production Companies: {', '.join(movie.get('production_companies', []))}
    Release Date: {movie.get('release_date', 'Unknown')}
    """
    
    return embedding_model.embed_query(text_data)

# Generate embeddings for all movies
movie_embeddings = []
total_movies = len(movies)
print(f"🔄 Generating embeddings for {total_movies} movies...")

for index, movie in enumerate(movies, start=1):
    embedding = generate_movie_embedding(movie)
    movie_embeddings.append({
        "id": movie.get("id", "Unknown"),
        "title": movie.get("title", "Unknown"),
        "embedding": embedding,
        "metadata": {
            "overview": movie.get("overview", "No overview available"),
            "genres": ", ".join(movie.get("genres", [])),
            "main_cast": ", ".join(movie.get("main_cast", [])),
            "director": movie.get("director", "Unknown"),
            "tagline": movie.get("tagline", "No tagline available"),
            "production_countries": ", ".join(movie.get("production_countries", [])),
            "keywords": ", ".join(movie.get("keywords", [])),
            "runtime": movie.get("runtime", "Unknown"),
            "production_companies": ", ".join(movie.get("production_companies", [])),
            "poster_path": movie.get("poster_path", None),  # Ensure poster_path is included
            "release_date": movie.get("release_date", "Unknown")  # Added release_date
        }
    })
    
    if index % 10 == 0 or index == total_movies:
        print(f"✅ Processed {index}/{total_movies} movies...")

# Save embeddings for later use
with open("movie_embeddings.json", "w") as f:
    json.dump(movie_embeddings, f, indent=4)

print("✅ Embeddings successfully generated and saved!")
```

#### Create `store_in_chromadb.py`:

✅ Loads precomputed movie embeddings from `movie_embeddings.json`.

✅ Initializes OpenAI's `text-embedding-ada-002` model (via LangChain) to generate mood embeddings.

✅ Stores mood embeddings in ChromaDB (`valid_moods` collection).

✅ Stores movie embeddings in ChromaDB (`movies` collection) with metadata.

✅ Deletes previous stored data before inserting new entries to ensure freshness.

```python
import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import OPENAI_API_KEY

# **Local storage for ChromaDB**
chroma_path = "chroma_db"
os.makedirs(chroma_path, exist_ok=True)

# **Initialize OpenAI Embeddings**
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# **Define valid moods**
valid_moods = [
    "happy", "joyful", "cheerful", "delighted", "gleeful", "content", "lighthearted", "beaming",
    "excited", "thrilled", "exhilarated", "ecstatic", "overjoyed", "pumped", "hyped", "giddy",
    "grateful", "thankful", "appreciative", "blessed", "fulfilled", "satisfied",
    "hopeful", "optimistic", "encouraged", "expectant", "inspired",
    "loving", "affectionate", "romantic", "caring", "devoted", "tender",
    "peaceful", "calm", "serene", "tranquil", "relaxed", "mellow",
    "proud", "accomplished", "confident", "empowered", "self-assured",
    "sad", "melancholic", "gloomy", "heartbroken", "dejected", "sorrowful",
    "lonely", "isolated", "abandoned", "rejected", "homesick", "neglected",
    "hopeless", "despairing", "pessimistic", "defeated", "discouraged",
    "bored", "indifferent", "unenthusiastic", "unstimulated", "listless",
    "guilty", "remorseful", "regretful", "ashamed", "embarrassed",
    "tired", "fatigued", "drained", "exhausted", "sluggish",
    "angry", "furious", "enraged", "irritated", "resentful", "bitter",
    "frustrated", "annoyed", "exasperated", "impatient", "aggravated",
    "jealous", "envious", "covetous", "possessive", "insecure",
    "disgusted", "repulsed", "revolted", "grossed out", "nauseated",
    "anxious", "nervous", "worried", "uneasy", "apprehensive", "jittery",
    "fearful", "terrified", "panicked", "paranoid", "tense", "alarmed",
    "overwhelmed", "stressed", "pressured", "frazzled", "overloaded",
    "surprised", "shocked", "amazed", "astonished", "stunned", "flabbergasted",
    "confused", "perplexed", "puzzled", "disoriented", "unsure", "uncertain",
    "indecisive", "conflicted", "hesitant", "torn", "ambivalent",
    "neutral", "indifferent", "meh", "emotionless", "numb",
    "bittersweet", "nostalgic", "wistful", "sentimental", "pensive",
    "thoughtful", "introspective", "brooding", "deep in thought"
]

# **Initialize ChromaDB for moods and delete old data before inserting new ones**
print("🔄 Resetting mood database before updating...")
mood_store = Chroma(
    persist_directory=chroma_path,
    embedding_function=embedding_model,
    collection_name="valid_moods"
)
mood_store.delete_collection()  # ✅ **Deletes old mood embeddings**
print("✅ Previous mood data cleared.")

# **Compute embeddings for valid moods**
print("🔄 Generating embeddings for valid moods...")
mood_texts = valid_moods
mood_embeddings = embedding_model.embed_documents(mood_texts)

# **Store new mood embeddings**
mood_store = Chroma(
    persist_directory=chroma_path,
    embedding_function=embedding_model,
    collection_name="valid_moods"
)
mood_store.add_texts(
    texts=mood_texts,
    metadatas=[{"mood": mood} for mood in valid_moods]
)
print(f"✅ Stored {len(valid_moods)} mood embeddings in ChromaDB ('valid_moods' collection).")

# **Load movie embeddings from JSON**
print("🔄 Loading movie metadata...")
with open("movie_embeddings.json", "r", encoding="utf-8") as f:
    movie_embeddings = json.load(f)

# **Helper function to process metadata safely**
def safe_join(value):
    """
    Safely converts lists to comma-separated strings for metadata storage.

    Args:
        value (list or str): The value to convert.

    Returns:
        str: A string representation of the value.
    """
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)

# **Initialize ChromaDB for movies and delete old data before inserting new ones**
print("🔄 Resetting movie database before updating...")
movie_store = Chroma(
    persist_directory=chroma_path,
    embedding_function=embedding_model,
    collection_name="movies"
)
movie_store.delete_collection()  # ✅ **Deletes old movie embeddings**
print("✅ Previous movie data cleared.")

# **Convert movie metadata into LangChain Document format**
print("🔄 Storing movies in ChromaDB...")
documents = []
total_movies = len(movie_embeddings)
for index, movie in enumerate(movie_embeddings, start=1):
    metadata = movie.get("metadata", {})
    doc_metadata = {
        "title": safe_join(movie.get("title", "Unknown")),
        "overview": safe_join(metadata.get("overview", "Unknown")),
        "genres": safe_join(metadata.get("genres", [])),
        "main_cast": safe_join(metadata.get("main_cast", [])),
        "director": safe_join(metadata.get("director", "Unknown")),
        "tagline": safe_join(metadata.get("tagline", "Unknown")),
        "production_countries": safe_join(metadata.get("production_countries", [])),
        "keywords": safe_join(metadata.get("keywords", [])),
        "runtime": safe_join(metadata.get("runtime", "Unknown")),
        "production_companies": safe_join(metadata.get("production_companies", [])),
        "poster_path": safe_join(metadata.get("poster_path", "Unknown")),  # Ensure poster_path is included
        "release_date": safe_join(metadata.get("release_date", "Unknown"))  # Ensure release_date is included
    }
    page_content = f"{doc_metadata['title']} {doc_metadata['overview']}"
    documents.append(Document(page_content=page_content, metadata=doc_metadata))
    if index % 10 == 0 or index == total_movies:
        print(f"✅ Processed {index}/{total_movies} movies...")
    # time.sleep(0.05)

# **Store new movie embeddings in ChromaDB**
movie_store = Chroma.from_documents(
    documents,
    embedding_model,
    persist_directory=chroma_path,
    collection_name="movies"
)
print(f"✅ ChromaDB successfully updated and stored at: {chroma_path}")
```

#### Create `app.py`:

✅ Accepts user input and converts it into an embedding.

✅ Matches the input mood with predefined mood embeddings using cosine similarity.

✅ Retrieves relevant movies based on the closest detected mood using ChromaDB.

✅ Generates AI-based movie recommendation explanations using OpenAI GPT-4o-mini.

✅ Displays movie metadata, posters, and explanations in a user-friendly interface.

```python
import streamlit as st
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import OPENAI_API_KEY
from PIL import Image
import requests
from io import BytesIO

def run_app():
    """
    Runs the Streamlit app for mood-based movie recommendations.
    
    Initializes:
    - OpenAI embedding model for vector representations.
    - ChromaDB for storing and retrieving movie embeddings.
    - OpenAI API client for generating explanations.
    
    The app accepts user input, detects mood, retrieves movie recommendations,
    and generates explanations dynamically.
    """
    # **Initialize OpenAI Embeddings & ChromaDB**
    chroma_path = "chroma_db"
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model,
        collection_name="movies"
    )

    mood_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model,
        collection_name="valid_moods"
    )

    client = OpenAI(api_key=OPENAI_API_KEY)

    # **Helper function to compute cosine similarity**
    def cosine_similarity(vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        
        Args:
            vec1 (np.array): First vector.
            vec2 (np.array): Second vector.

        Returns:
            float: Cosine similarity score between -1 and 1.
        """
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

    # **Function to get the closest mood**
    def get_top_mood(user_input):
        """
        Identifies the closest predefined mood to the user input.

        Args:
            user_input (str): User's mood description.

        Returns:
            list: Top 3 detected moods or None if no match is found.
        """
        user_mood_vector = embedding_model.embed_query(user_input)
        mood_retriever = mood_store.as_retriever(search_kwargs={"k": 5})
        valid_mood_results = mood_retriever.invoke(user_input)

        similarities = []
        unique_moods = set()
        
        for mood in valid_mood_results:
            mood_name = mood.metadata.get("mood", "").lower()
            if mood_name not in unique_moods:  # Ensure uniqueness
                mood_vector = embedding_model.embed_query(mood_name)
                similarity_score = cosine_similarity(user_mood_vector, mood_vector)
                similarities.append((mood_name, similarity_score))
                unique_moods.add(mood_name)

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

        if not similarities or similarities[0][1] < 0.8:
            return None  # No valid moods found

        return [mood[0] for mood in similarities]  # Return top 3 distinct moods

    # **Function to get unique movie recommendations**
    def get_movie_recommendations(detect_moods):
        """
        Retrieves unique movie recommendations from ChromaDB.

        Args:
            detect_moods (str): The detected mood used for querying.

        Returns:
            list: Top 3 unique recommended movies based on mood.
        """
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        results = retriever.invoke(detect_moods)

        unique_movies = {}
        for movie in results:
            title = movie.metadata.get("title", "Unknown")
            if title not in unique_movies:
                unique_movies[title] = movie

        return list(unique_movies.values())[:3]  # Return only top 3 unique movies

    # **Function to generate LLM-based explanation**
    def generate_explanation(detect_moods, user_input, movie):
        """
        Retrieves unique movie recommendations from ChromaDB.

        Args:
            detect_moods (str): The detected mood used for querying.

        Returns:
            list: Top 3 unique recommended movies based on mood.
        """
        movie_description = (
            f"{movie.metadata.get('title', 'Unknown')} ({movie.metadata.get('genres', 'Unknown')})\n"
            f"Overview: {movie.metadata.get('overview', 'No overview available')}\n"
        )

        explanation_prompt = f"""
        A user said {user_input} and might be feeling {detect_moods}. Based on what they said and their moods, here is the recommended movie:

        {movie_description}

        Generate a friendly, engaging movie recommendation explanation that highlights why this movie might be a good fit for the user.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful movie recommendation assistant."},
                {"role": "user", "content": explanation_prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    # **Function to fetch movie poster**
    def get_movie_poster(url):
        """
        Fetches the movie poster from a given URL.

        Args:
            url (str): The movie poster URL.

        Returns:
            PIL.Image or None: Image object if successful, otherwise None.
        """
        if not url or not url.startswith("http"):
            return None  # Invalid or missing URL

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image
        except requests.exceptions.RequestException:
            return None  # Return None if request fails

        return None  # Default fallback if nothing works

    # **Streamlit UI**
    st.title("🎬 CineMood: Get Mood-Based Trending Movies! ⚡")

    st.write("Enter how you're feeling right now, and we'll recommend the best movies for you!")

    user_input = st.text_input("How are you feeling now?", placeholder="E.g., happy, nostalgic, adventurous, I am missing my lovely daughter, etc.")

    if st.button("🎥 Recommend Movies"):
        st.write(f"ChromaDB Movies Collection Size: {vector_store._collection.count()}")
        st.write(f"ChromaDB Moods Collection Size: {mood_store._collection.count()}")

        if not user_input:
            st.warning("⚠️ Please enter how you're feeling.")
        else:
            with st.spinner("Detecting your mood and finding the best movies for you..."):
                detect_moods = get_top_mood(user_input)

                if not detect_moods:
                    st.warning("⚠️ Oh, I’m not quite sure I caught that mood! Could you share how you're feeling in another way? I'd love to find the perfect movie for you!")
                else:
                    # st.success(f"🤖 Detected Moods: {', '.join(detect_moods)}")

                    # Use only the first mood for movie recommendations
                    top_movies = get_movie_recommendations(detect_moods[0])

                    if not top_movies:
                        st.error("❌ No relevant movies found for your mood.")
                    else:
                        # explanation = generate_explanation(detect_moods[0], user_input, top_movies)

                        st.markdown("## 🎬 **Top 3 Movie Recommendations**")

                        for i, movie in enumerate(top_movies):
                            metadata = movie.metadata
                            st.subheader(f"{i+1}. {metadata.get('title', 'Unknown')} ({metadata.get('genres', 'Unknown')})")
                            st.write(f"**📅 Release Date:** {metadata.get('release_date', 'Unknown')}")
                            st.write(f"**🎭 Cast:** {metadata.get('main_cast', 'Unknown')}")
                            st.write(f"**🎬 Director:** {metadata.get('director', 'Unknown')}")
                            st.write(f"**🏷️ Tagline:** {metadata.get('tagline', 'Unknown')}")
                            st.write(f"**🌍 Country:** {metadata.get('production_countries', 'Unknown')}")
                            st.write(f"**🏢 Production Company:** {metadata.get('production_companies', 'Unknown')}")
                            st.write(f"**⏳ Runtime:** {metadata.get('runtime', 'Unknown')} min")

                            poster_url = metadata.get('poster_path', '')
                            if poster_url:
                                image = get_movie_poster(poster_url)
                                if image:
                                    st.image(image, width=200)
                                else:
                                    st.warning("⚠️ Poster not available.")
                            
                            explanation = generate_explanation(detect_moods[0], user_input, movie)
                            st.write(explanation)

                            st.write("---")

    st.markdown("**Made by [Thanh Tung Vu](https://thanhtungvudata.github.io/)**")

if __name__ == "__main__":
    run_app()
                        
```

## **Results & Improvements**
With this RAG-based approach, the **movie recommendations are more accurate, scalable, and explainable**. The app now:

✅ Works with **large-scale datasets** stored in ChromaDB.

✅ Validates moods **more effectively** using **vector similarity**.

✅ Reduces **hallucination by grounding explanations in real movie metadata**.

✅ Provides **dynamic explanations using LLM** while maintaining factual correctness.

## **🎉 Results and Live Demo**

The final web app delivers mood-based movie recommendations in just a second, with fresh content every week. 

You can try it here:👉 [CineMood Live App on Hugging Face Spaces](https://huggingface.co/spaces/thanhtungvudata/CineMoodv3)

<div style="text-align: center;">
    <h3>🎬 Try CineMoodv3 Now!</h3>
    <iframe
        src="https://thanhtungvudata-cinemoodv3.hf.space"
        width="100%"
        height="600"
        style="border: none; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);"
        allowfullscreen>
    </iframe>
</div>

## **Tech Stack**
1. Data Storage & Retrieval
- Vector Database: ChromaDB (stores and retrieves movie embeddings efficiently for fast and scalable similarity search)
2. Embedding & Retrieval-Augmented Generation (RAG) Pipeline
- LLM Model: OpenAI GPT-4o-mini (generates responses based on retrieved movie data)
- Embedding Model: OpenAI’s text-embedding-ada-002 (encodes movie metadata and user input into a high-dimensional vector space)
- Similarity Search: Cosine similarity with ChromaDB (retrieves the most relevant movies based on user queries)
- Document Processing: LangChain (manages RAG-based retrieval, query execution, and data flow)
3. Web Application (Self-Contained, No Separate Backend Needed)
- Framework: Streamlit (builds an interactive web interface and processes user inputs)
- Programming Language: Python (integrates Streamlit, retrieval logic, and ML models)
- UI Components: Streamlit’s built-in widgets (for creating a user-friendly interface)
4. Model Deployment & Infrastructure
- Web App Hosting: Hugging Face Spaces (hosts the Streamlit app for public access)
- Database Updates: GitHub Actions (automates weekly ChromaDB updates to keep recommendations fresh)
- Containerization: Docker (optional, for packaging and deploying the app efficiently on HF Spaces)

### 🔗 Why Use LangChain?
LangChain is a framework for building LLM-powered applications, and in this project, it helps with:

1. Embedding Generation:
- The project uses OpenAI's text-embedding-ada-002 model via LangChain to convert movie metadata and user mood descriptions into high-dimensional vector embeddings.
- LangChain provides a simple interface (OpenAIEmbeddings) to generate these embeddings.
2. Vector Retrieval:
- LangChain integrates ChromaDB as a vector store, allowing efficient semantic search for movies and moods.
- Instead of using basic keyword matching, LangChain allows embedding-based similarity search to retrieve the best recommendations.
3. Query Execution & Data Flow:
- The project needs to retrieve and rank movies based on mood similarity.
- LangChain’s Chroma interface simplifies managing retrieval pipelines for structured responses.

Also, why LangChain over other LLM orchestration frameworks?

✅ Provides OpenAI API integration out of the box (unlike Sentence-Transformers (SBERT) + FAISS (Facebook AI Similarity Search)).

✅ Abstracts away retrieval complexity with simple .as_retriever().

✅ Flexible: Works with multiple vector databases like Chroma, Pinecone, or FAISS.


### 💾 Why Use ChromaDB?
ChromaDB is an open-source vector database designed for fast similarity search. In this project, it enables:

1. Efficient Storage of Embeddings:
- The system stores precomputed embeddings for movies and valid moods in ChromaDB.
- Storing embeddings in a structured way enables fast and scalable retrieval.
2. Similarity-Based Search (Cosine Similarity):
- When a user enters a mood, the app embeds their input and finds the most similar stored moods in ChromaDB.
- This avoids manual keyword mapping and allows more accurate mood matching.
3. Fast Retrieval at Scale:
- Unlike relational databases (PostgreSQL, MySQL), which are not optimized for vector search, ChromaDB enables instantaneous nearest-neighbor searches.
- This ensures the system can scale to thousands/millions of movies without performance degradation.

Also, why ChromaDB over other vector databases?

✅ Lightweight & local-friendly (unlike Pinecone).

✅ Easiest to integrate with LangChain (native support).

✅ Free & open-source (no extra cloud costs).


## **Conclusion & Next Steps**
By integrating **RAG with OpenAI embeddings and ChromaDB**, the **mood-based movie recommendation app** has become **more scalable, reliable, and explainable**. Moving forward, possible improvements include:

🔹 **Fine-tuning threshold values for similarity scores**.

🔹 **Expanding the movie dataset** to include more diverse genres and countries.

🔹 **Integrating user feedback to refine future recommendations**.

🔹 **Enhancing explanation generation by incorporating more structured metadata.**

🚀 **Let me know what you think!**

---
📌 **Stay tuned for more updates on AI-powered movie recommendations!** 🎬✨

The code of this project is available [here](https://github.com/thanhtungvudata/cinemood2/tree/main/CineMood%20v2). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
