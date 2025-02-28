---
title: "Handling Validation and Hallucination Issues in LLM for a Mood-Based Movie Recommendation App"
date: 2025-02-28
image: /assets/images/cinemood2_overview.png
categories:
  - Data Science Projects
tags:
  - Data Science
  - AI Agent
  - ML Engineer
---
Large Language Models (LLMs) are powerful tools for natural language understanding, but they can introduce issues such as validation errors and hallucination. In our Mood-Based Movie Recommendation App [CineMood2](https://thanhtungvudata.github.io/data%20science%20projects/cinemood2-update/), these issues previously caused unreliable mood detection and incorrect movie suggestions. This blog post explores the challenges in the older version of our code and the improvements in the new version to ensure accurate recommendations. 

<img src="/assets/images/cinemood2_overview2.png" alt="CineMood2" width="600">

---

## The Problems in the Old Code
### 1. **Validation Issues**
Validation ensures that the input data (user moods) and outputs (movie recommendations) are correct and meaningful. The old code had the following problems:
- The `detect_mood()` function **did not validate** if the extracted moods were within a predefined list of valid moods.
- If the LLM returned fewer than three moods, the app did not have proper fallback mechanisms.
- If the input text was not a mood description (e.g., "What time is it?"), the LLM could still return arbitrary moods.

### 2. **Hallucination Issues**
Hallucination occurs when an LLM generates responses that are **plausible but incorrect**. The old implementation had:
- No strict **predefined set of moods**, allowing the model to invent moods that were not relevant.
- No validation on LLM-generated JSON outputs, potentially causing errors in movie ranking.
- No fallback strategy when the LLM failed or returned unexpected responses.

---

## Step-by-Step Fixes in the New Code

### **1. Improved Validation in `detect_mood()`**
#### **Old Implementation**
- The previous version simply extracted three words from user input without checking if they were valid moods.
- No clear handling of completely unrelated inputs.

#### **New Implementation**
✅ **Predefined Mood List**: The updated version ensures that the detected moods belong to a predefined set of **VALID_MOOD_WORDS**.

✅ **Invalid Input Handling**: If the input does not describe a mood, the function returns `["invalid"]`, prompting the user for a clearer response.

✅ **Fallback to Neutral Moods**: If fewer than three moods are detected, it ensures that exactly **three moods** are returned by adding "neutral" as necessary.

### **2. Preventing Hallucination in `detect_mood()`**
#### **Old Implementation**
- Relied entirely on GPT output without verifying if the moods were meaningful.

#### **New Implementation**
✅ **Strictly Enforced Output Format**: GPT is prompted to return **only** a valid JSON response with **exactly three moods**.

✅ **Mapping Unknown Moods**: If an unknown mood appears, the function remaps it to the closest valid mood or defaults to neutral.

---
## 🛠 **Technical Details:**

- Revised `llm.py`:
```python
import json
import openai
from config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Updated Mood List
VALID_MOOD_WORDS = set([
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
])

# ✅ Function to Map Detected Moods Using GPT
def map_to_valid_mood(mood_words):
    """
    Uses GPT to determine the closest valid moods from `VALID_MOOD_WORDS`.
    Ensures that:
    - The result contains **exactly 3 unique** moods.
    - All moods are from `VALID_MOOD_WORDS`.
    - If fewer than 3 moods are returned, "neutral" is added.
    """

    valid_moods_string = ", ".join(VALID_MOOD_WORDS)

    prompt = f"""
    You are an expert in understanding human emotions.
    Your task is to map the detected moods "{', '.join(mood_words)}" to the **three closest** valid moods from the predefined list below:

    {valid_moods_string}

    **Rules:**
    1. Select exactly **3 unique** moods from the list.
    2. If a mood is unrelated to any in the list, replace it with **"neutral"**.
    3. **Return ONLY a comma-separated list of 3 moods, no extra text.**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )

        # ✅ Extract and clean GPT response
        mapped_moods = response.choices[0].message.content.strip().lower()

        # ✅ Handle cases where GPT incorrectly formats the output
        mapped_moods = mapped_moods.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        mapped_moods = mapped_moods.split(", ")
        mapped_moods = [mood.strip() for mood in mapped_moods if mood in VALID_MOOD_WORDS]

        # ✅ Ensure exactly 3 unique moods
        unique_moods = list(dict.fromkeys(mapped_moods))  # Remove duplicates while keeping order

        # ✅ Fill with "neutral" if fewer than 3 moods are returned
        while len(unique_moods) < 3:
            unique_moods.append("neutral")

        return unique_moods[:3]  # Always return a single flat list of 3 moods

    except Exception as e:
        print(f"⚠️ Error in mapping mood: {e}")
        return ["neutral", "neutral", "neutral"]  # Default in case of error


# ✅ Function to Detect Mood
def detect_mood(user_input):
    """
    Detects mood from user input:
    - Extracts key words from the user's input.
    - Uses GPT to map detected moods to `VALID_MOOD_WORDS`.
    - Returns ['invalid'] for non-emotional input.
    """
    valid_moods_string = ", ".join(VALID_MOOD_WORDS)

    prompt = f"""
    Analyze the user input {user_input} and:
    If the input {user_input} has any words that are related to at least one of these words {valid_moods_string} or related to the nouns of {valid_moods_string}, return output as a **valid JSON object** with:
        - "detected_moods": **Always a list of exactly 3 different mood words that best describe the user's input {user_input}** 
        - "extracted_words": The key words you identified in the user's input {user_input}.

    If input {user_input} is **related to a mood** but not related to at least one of these words {valid_moods_string} or related to the nouns of {valid_moods_string}, return output as a **valid JSON object** with:
        - "detected_moods": **Always a list of exactly 3 different mood words that best describe the user's input {user_input}** 
        - "extracted_words": The key words you identified in the user's input {user_input}.

    If the input {user_input} is **COMPLETELY NOT related to a mood** (e.g., "What time is it?"), return output as a **valid JSON object** with:
        - "detected_moods": ["invalid"] 
        - "extracted_words": [].

    Example:
    User input: "I feel a bit lost and unsure what to do."
    Response:
    {{"detected_moods": ["melancholic", "uncertain", "conflicted"], "extracted_words": ["lost", "unsure"]}}

    User input: "What time is it?"
    Response:
    {{"detected_moods": ["invalid"], "extracted_words": []}}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )

        json_response = json.loads(response.choices[0].message.content.strip())
        detected_moods = json_response.get("detected_moods", [])
        extracted_words = json_response.get("extracted_words", [])

        if detected_moods == ["invalid"]:
            return ["invalid"], [], []

        # ✅ Separate known moods from unknown moods
        known_moods = [mood for mood in detected_moods if mood in VALID_MOOD_WORDS]
        unknown_moods = [mood for mood in detected_moods if mood not in VALID_MOOD_WORDS]

        # ✅ Map unknown moods to valid moods using GPT only if necessary
        mapped_moods = map_to_valid_mood(unknown_moods) if unknown_moods else []

        # ✅ Combine known and mapped moods, ensuring 3 unique moods
        final_moods = list(dict.fromkeys(known_moods + mapped_moods))  # Remove duplicates

        while len(final_moods) < 3:
            final_moods.append("neutral")

        return final_moods[:3], extracted_words, detected_moods

    except json.JSONDecodeError:
        print("⚠️ Error: GPT returned invalid JSON.")
        return ["neutral", "neutral", "neutral"], [], []
    
    except Exception as e:
        print(f"⚠️ Error in detect_mood: {e}")
        return ["neutral", "neutral", "neutral"], [], []


def get_movies_by_mood(mood_words, movies):
    """
    Uses GPT to rank movies based on detected moods or extracted words.
    ✅ If mood is ["neutral", "neutral", "neutral"], match using extracted words.
    ✅ Otherwise, rank movies based on emotional relevance.
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
    
    The user’s detected moods: {", ".join(mood_words)}.
    
    Below are movie descriptions:
    {movie_descriptions}
    
    Select the top 3 movies that best match this mood or extracted words and provide a brief explanation (1-2 sentences) for each.
    Respond strictly in JSON format:
    [
        {{"index": 1, "match_reason": "Explanation for movie 1"}},
        {{"index": 2, "match_reason": "Explanation for movie 2"}},
        {{"index": 3, "match_reason": "Explanation for movie 3"}}
    ]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        json_response = json.loads(response.choices[0].message.content.strip())

        matched_movies = []
        for entry in json_response:
            index = entry["index"] - 1
            explanation = entry.get("match_reason", "Trending movie recommendation.")
            if 0 <= index < len(movies):
                matched_movie = movies[index]
                matched_movie["match_reason"] = explanation
                matched_movies.append(matched_movie)

        return matched_movies

    except Exception as e:
        print(f"⚠️ Error ranking movies: {e}")
        return movies[:3]  # Default to trending movies
```

- Revised `app.py`:
```python
import streamlit as st
from llm import detect_mood, get_movies_by_mood
from tmdb_api import fetch_movies

def run_app():
    """
    Runs the Streamlit app for mood-based movie recommendations.
    Handles cases where detect_mood() returns only 2 values.
    """
    st.set_page_config(
        page_title="🎬 Mood-Based Movie Recommendation", 
        layout="centered"
    )

    st.title("🎬 CineMood: Get Mood-Based Trending Movies! ⚡")

    user_mood = st.text_area(
        "💬 How do you feel right now?",
        st.session_state.get("user_mood", ""),
        height=100
    )

    if st.button("Find Movies"):
        if user_mood.strip():
            with st.spinner("🔍 Analyzing your mood..."):
                valid_moods, extracted_words, detected_moods = detect_mood(user_mood)

            if valid_moods == ["invalid"]:
                st.warning("⚠️ That doesn't look like a mood. Please describe how you're feeling.")
            else:
                st.success(f"🤖 AI Detected Moods: {', '.join(valid_moods).title()}")

                with st.spinner("🎥 Fetching movies and ranking matches..."):
                    movies = fetch_movies(60)

                    if valid_moods == ["neutral", "neutral", "neutral"]:
                        st.info("🎭 We couldn't be sure about your moods, so let us guess. Here are some trending movies you might enjoy!")

                        if extracted_words:
                            st.write(f"🔍 AI detected these key words from your input: **{', '.join(extracted_words)}**")
                            recommended_movies = get_movies_by_mood(extracted_words, movies)
                        else:
                            recommended_movies = movies[:3]
                    else:
                        recommended_movies = get_movies_by_mood(valid_moods, movies)

                if recommended_movies:
                    for movie in recommended_movies:
                        st.subheader(movie["title"])
                        st.write(f"📅 Release Date: {movie['release_date']}")
                        st.write(f"🎭 Match Reason: {movie.get('match_reason', 'Trending movie recommendation.')}")
                        if movie["poster"]:
                            st.image(movie["poster"], width=200)
                        st.write(f"📜 Overview: {movie['overview']}")
                        st.markdown("---")
                else:
                    st.warning("⚠️ No suitable movie recommendations found.")
        else:
            st.warning("⚠️ Please enter how you feel to get movie recommendations.")

    st.markdown("**Made by [Thanh Tung Vu](https://thanhtungvudata.github.io/)**")

if __name__ == "__main__":
    run_app()

```

- File `tmdb_api.py` is the same.

---

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
By implementing **strict validation** and **hallucination prevention techniques**, our Mood-Based Movie Recommendation App now provides **reliable** and **accurate** movie suggestions. The updated version ensures:
- ✅ Proper validation of moods and user input.
- ✅ Preventing hallucination by strictly defining mood categories.

These enhancements significantly improve the **user experience** and **trustworthiness** of AI-powered recommendations!

The code of this project is available [here](https://github.com/thanhtungvudata/mood_based_trending_movie_recommendation). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
