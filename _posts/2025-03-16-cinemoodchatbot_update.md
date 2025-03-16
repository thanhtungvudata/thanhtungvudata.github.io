---
title: "Building a Chatbot for Mood-Based Trending Movie Recommendation"
date: 2025-03-16
image: /assets/images/cinemood2_overview.png
categories:
  - Data Science Projects
tags:
  - Data Science
  - AI Agent
  - ML Engineer
  - RAG
---
In my previous [post](https://thanhtungvudata.github.io/data%20science%20projects/cinemood3-udpate/), I showed how to build a RAG (Retrieval-Augmented Generation) mood-based trending movie recommendation app. In this post, we dive into how to build a web-based chatbot that recommends trending movies based on users' moods and preferences. This latest update overcomes previous limitations by introducing **conversational interactions** and **advanced mood detection** using the **GPT-4o-mini** model with careful **prompt engineering**.

<img src="/assets/images/cinemoodchatbot_overview.png" alt="CineMood Chatbot" width="600">


## 🎉 What’s New: Improvements Over the Previous Version

The earlier version of our movie recommendation app used a **Large Language Model (LLM)** to analyze a fixed set of trending movies. However, it had significant limitations:

❌ **Static Interaction:** Users could input their mood, but the app lacked dynamic back-and-forth conversation to refine recommendations.

❌ **Limited Mood Validation:** Mood detection relied on simple semantic search, making it vulnerable to misinterpretation when users mixed emotional descriptions with additional preferences.

💡 **New Enhancements:**

✅ **Conversational Chatbot Integration:** The chatbot now maintains context and refines recommendations dynamically, engaging users through questions.

✅ **Advanced Mood Detection:** Instead of basic keyword matching, **GPT-4o-mini** carefully extracts the dominant mood from complex user inputs using structured **prompt engineering**.

✅ **Scalable & Interactive:** The chatbot interacts with users, validates moods against a **vector database (ChromaDB)**, and retrieves the most relevant movies efficiently.

## 🤖 Overcoming Mood Detection Challenges

Detecting moods accurately from user inputs is **not straightforward**. Users often provide **long, nuanced texts** that mix **emotions, personal stories, and preferences**. Here are the challenges we tackled:

🔹 **Complex Inputs:** A user might say, *"I’m feeling nostalgic yet excited because I miss my old movie nights."* This blends multiple emotions, requiring intelligent extraction of the **primary mood**.

🔹 **Semantic Ambiguity:** Some moods are context-dependent, requiring additional validation. *For example, 'bittersweet' could mean happy or sad, depending on the situation.*

🔹 **Prompt Engineering with GPT-4o-mini:** Our chatbot **filters out extraneous information**, ensuring that it extracts only the **core mood** from the list of valid moods.

## 🎬 Key Features of the Chatbot

⭐ **Conversational Flow:** The chatbot **remembers past interactions**, allowing it to clarify and refine recommendations dynamically.

🧠 **Intelligent Mood Validation:** The chatbot **embeds user input** and compares it against a stored **vector database of moods**. If uncertain, it asks the user for **clarification**.

📀 **Scalable Movie Retrieval:** Unlike the previous version, which **processed a fixed dataset**, the new chatbot **searches dynamically** through a **large collection** using **semantic similarity**.

📜 **Fact-Based Explanations:** Instead of generating random explanations, **GPT-4o-mini** **grounds its responses** in real metadata like **movie titles, summaries, and cast details**.

🖥️ **User-Friendly Interface:** Built with **Streamlit**, the chatbot provides **interactive conversations, mood tracking, and personalized recommendations** with **movie posters, genres, and trailers**.

By integrating RAG with advanced mood detection and leveraging robust tools like ChromaDB and GPT-4o-mini, our chatbot has evolved into a scalable, reliable, and highly interactive movie recommendation system. The combination of:
- **Careful prompt engineering for precise mood detection,**
- **Efficient vector-based retrieval for trending movie recommendation,** and
- **A user-friendly conversational interface**

ensures that users receive tailored movie recommendations that truly resonate with their current mood.

## **Technical Details**
Compared to the previous post, we only need to update the file `app.py` for the chatbot:

✅ Setup OpenAI Embeddings, ChromaDB, and GPT-4o-mini LLM for conversational responses and movie recommendations.

✅ Load and store movie metadata in ChromaDB for efficient vector-based retrieval.

✅ Maintain conversation context using ConversationBufferMemory, allowing users to refine their preferences dynamically.

✅ Define valid moods and movie genres to help extract relevant information from user input.

✅ Extract moods, genres, runtime, release date, cast, director, production country, and production company from user input using carefully crafted prompts with GPT-4o-mini.

✅ Validate user moods by checking extracted mood words against a predefined list.

✅ Detect runtime constraints (e.g., "less than 120 minutes") using regex parsing and LLM-based extraction.

✅ Detect release date conditions (e.g., "movies before 2010") and apply filtering accordingly.

✅ Retrieve relevant movie recommendations based on the extracted criteria using ChromaDB's vector search.

✅ Generate natural language explanations for the recommended movies using GPT-4o-mini.

✅ Provide a conversational interface in Streamlit, keeping track of user preferences and iterating through clarification questions if needed.

✅ Handle user requests to restart preferences ("fresh start") and allow dynamic modifications.

✅ Display movie posters dynamically alongside recommendations.

✅ Provide follow-up interactions, prompting users to refine their selections or reset preferences.

This chatbot effectively combines RAG and conversational AI to deliver personalized movie recommendations based on user mood and preferences.

```python
import streamlit as st
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import OPENAI_API_KEY
import requests
import base64

def run_app():

    # -------------------------------------------
    # Setup: OpenAI embeddings, local DB, and LLM
    # -------------------------------------------
    chroma_path = "chroma_db"
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model,
        collection_name="movies"
    )

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

    genres_list = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
        "Music", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"
    ]

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    chatbot_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    # -------------------------------------------
    # Helper functions
    # -------------------------------------------

    def generate_conversational_response(user_input):
        """
        Generates a short, friendly conversation-style reply WITHOUT giving direct movie recommendations.
        """
        prompt = (
            "You are a friendly assistant. The user might mention mood or movie preferences. "
            "Do NOT provide any specific movie recommendations. Just acknowledge.\n\n"
            f"User: {user_input}\n\n"
            "Your short, friendly response (no movie recommendations):"
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def get_user_mood(user_input, valid_moods, genres_list):
        """
        Attempts to find exactly one valid mood in the user_input.
        If found, returns the mood (lowercased). Otherwise returns None.
        """
        prompt = (
            "You have a list of valid moods:\n"
            f"{', '.join(valid_moods)}\n\n"
            "You have a list of genres:\n"
            f"{', '.join(genres_list)}\n\n"
            f"User input: {user_input}\n"
            "Ignore any words of the user input related to genres or other phrases." 
            "Extract exactly one mood from the user input if it is related in the valid mood list."
            "If no valid mood is found, return 'invalid'.\n\n"
            "No explanation needed, just extract the mood word as instructed."
        )
        response = llm.invoke(prompt)
        mood_candidate = response.content.strip().lower() if hasattr(response, "content") else response.strip().lower()
        if mood_candidate in valid_moods:
            return mood_candidate
        return None

    def detect_genre(user_input, valid_moods, genres_list):
        prompt = (
            "You have a list of valid moods:\n"
            f"{', '.join(valid_moods)}\n\n"
            "You have a list of genres:\n"
            f"{', '.join(genres_list)}\n\n"
            f"User input: {user_input}\n"
            "Ignore any words of the user input related to mood or other phrases." 
            "Extract exactly one word from the user input if this word is related to the genre list."
            "If no valid mood is found, return 'None'.\n\n"
            "No explanation needed, just extract the genre as instructed."
        )
        response = llm.invoke(prompt)
        genre = response.content.strip() if hasattr(response, "content") else response.strip()
        return genre if genre.lower() != "none" else None

    def detect_runtime(user_input):
        """
        Detect runtime expressions like:
        - "less than 120 minutes" -> "< 120"
        - "under 2 hours" -> "< 120"
        - "over 90 minutes" -> "> 90"
        - "exactly 100 minutes" -> "= 100"
        If no mention of runtime is found, return None.
        """
        prompt = (
            f"User input: {user_input}\n"
            f"Return '< X' if the user input says something like 'less than X minutes' or 'under X minutes'.\n"
            f"Return '> X' if the user input says something like 'more than X minutes' or 'over X minutes'.\n"
            f"Return '= X' if the user input says something like 'exactly X minutes' or 'X minutes long'.\n"
            "If no runtime mention is found, return 'None'.\n\n"
            "The output should be in the format of '< X', '> X', '= X' or 'None', without any apostrophe. "
            "No explanation needed, just extract the runtime condition as instructed."
        )
        response = llm.invoke(prompt)
        runtime_candidate = response.content.strip() if hasattr(response, "content") else response.strip()
        pattern = r'^[<>]=?\s*\d+$'
        if re.match(pattern, runtime_candidate):
            return runtime_candidate
        return None if runtime_candidate.lower() == "none" else runtime_candidate

    def detect_release_date(user_input):
        """
        Uses GPT-4o-mini to extract a release date condition from the user input.
        Return "< YYYY" if the input mentions a release date before a year,
        "> YYYY" if after a year, or "= YYYY" if it specifies a particular year.
        If no release date mention is found, return 'None'.
        """
        prompt = (
            f"User input: {user_input}\n"
            "Return only '< YYYY' if the user input says something like 'before YYYY' or 'prior to YYYY'.\n"
            "Return only '> YYYY' if the user input says something like 'after YYYY' or 'post YYYY'.\n"
            "Return only '= YYYY' if the user input says something like 'in YYYY' or 'released in YYYY'.\n"
            "If no release date mention is found, return 'None'.\n\n"
            "The output should be in the format of '< YYYY', '> YYYY', '= YYYY' or 'None', without any apostrophe."
            "No explanation needed, just extract the release date condition as instructed."
        )
        response = llm.invoke(prompt)
        release_date_candidate = response.content.strip() if hasattr(response, "content") else response.strip()
        pattern = r'^[<>]=?\s*\d{4}$'
        if re.match(pattern, release_date_candidate):
            return release_date_candidate
        return None if release_date_candidate.lower() == "none" else release_date_candidate

    def detect_cast(user_input):
        prompt = (
            f"User input: {user_input}"
            "Extract the main cast names mentioned in the user input. "
            "If no cast is mentioned, reply with 'None'.\n\n"
            "No explanation needed, just extract the main cast names as instructed."
        )
        response = llm.invoke(prompt)
        cast = response.content.strip() if hasattr(response, "content") else response.strip()
        return cast if cast.lower() != "none" else None

    def detect_director(user_input):
        prompt = (
            f"User input: {user_input}"
            "Extract the director name mentioned in the user input. "
            "If no director is mentioned, reply with 'None'.\n\n"
            "No explanation needed, just extract the director name as instructed."
        )
        response = llm.invoke(prompt)
        director = response.content.strip() if hasattr(response, "content") else response.strip()
        return director if director.lower() != "none" else None

    def detect_production_country(user_input):
        prompt = (
            f"User input: {user_input}"
            "Return only the name of the production country mentioned in the user input. "
            "If no production country is mentioned, reply with 'None'.\n\n"
            "No explanation needed, just extract the production country as instructed."
        )
        response = llm.invoke(prompt)
        country = response.content.strip() if hasattr(response, "content") else response.strip()
        return country if country.lower() != "none" else None

    def detect_production_company(user_input):
        prompt = (
            f"User input: {user_input}"
            "Return only the name of the production company mentioned in the user input. "
            "If no production company is mentioned, reply with 'None'.\n\n"
            "No explanation needed, just extract the production company as instructed."
        )
        response = llm.invoke(prompt)
        company = response.content.strip() if hasattr(response, "content") else response.strip()
        return company if company.lower() != "none" else None

    def get_movie_recommendations(mood, genre=None, release_date=None, cast=None, director=None, 
                                production_country=None, production_company=None, runtime=None):
        """
        Retrieves top 3 movies from local DB that match all specified criteria.
        Also handles runtime expressions like "< 120", "> 120", "= 120" and release date conditions like "< 2000".
        """
        query_components = []
        if mood and mood.lower() != "not specified":
            query_components.append(mood)
        if genre and genre.lower() != "not specified":
            query_components.append(genre)
        if release_date and release_date.lower() != "not specified":
            query_components.append(release_date)
        if cast and cast.lower() != "not specified":
            query_components.append(cast)
        if director and director.lower() != "not specified":
            query_components.append(director)
        if production_country and production_country.lower() != "not specified":
            query_components.append(production_country)
        if production_company and production_company.lower() != "not specified":
            query_components.append(production_company)
        if runtime and runtime.lower() != "not specified":
            query_components.append(runtime)

        query = " ".join(query_components).strip()
        if not query:
            return []

        retriever_local = vector_store.as_retriever(search_kwargs={"k": 100})
        results = retriever_local.invoke(query)
        unique_movies = {}

        # Parse runtime operator if any
        op_runtime = None
        rt_value = None
        if runtime:
            match = re.match(r'([<>]=?|\=)\s*(\d+)', runtime)
            if match:
                op_runtime, rt_str = match.groups()
                rt_value = int(rt_str)

        # Parse release date operator if any
        op_release = None
        rd_value = None
        if release_date:
            match = re.match(r'([<>]=?|\=)\s*(\d{4})', release_date)
            if match:
                op_release, rd_str = match.groups()
                rd_value = int(rd_str)

        for movie in results:
            title = movie.metadata.get("title", "Unknown")
            movie_genres = movie.metadata.get("genres", "").split(", ")
            movie_release = movie.metadata.get("release_date", "")
            movie_cast = movie.metadata.get("main_cast", "")
            movie_director = movie.metadata.get("director", "")
            movie_country = movie.metadata.get("production_countries", "")
            movie_company = movie.metadata.get("production_companies", "")
            movie_runtime_str = movie.metadata.get("runtime", None)

            # Attempt to parse runtime as int
            movie_runtime_val = None
            if movie_runtime_str is not None:
                try:
                    movie_runtime_val = int(movie_runtime_str)
                except ValueError:
                    movie_runtime_val = None

            # Attempt to parse release date year as int (first 4 digits)
            movie_release_val = None
            if movie_release:
                try:
                    movie_release_val = int(movie_release[:4])
                except ValueError:
                    movie_release_val = None

            if genre and genre.lower() != "not specified":
                if genre not in movie_genres:
                    continue
            if cast and cast.lower() != "not specified":
                if cast not in movie_cast:
                    continue
            if director and director.lower() != "not specified":
                if director not in movie_director:
                    continue
            if production_country and production_country.lower() != "not specified":
                if production_country not in movie_country:
                    continue
            if production_company and production_company.lower() != "not specified":
                if production_company not in movie_company:
                    continue

            # Handle runtime filtering
            if op_runtime and rt_value is not None and movie_runtime_val is not None:
                if op_runtime == "<":
                    if not (movie_runtime_val < rt_value):
                        continue
                elif op_runtime == "<=":
                    if not (movie_runtime_val <= rt_value):
                        continue
                elif op_runtime == ">":
                    if not (movie_runtime_val > rt_value):
                        continue
                elif op_runtime == ">=":
                    if not (movie_runtime_val >= rt_value):
                        continue
                elif op_runtime in ["=", "=="]:
                    if movie_runtime_val != rt_value:
                        continue

            # Handle release date filtering
            if op_release and rd_value is not None and movie_release_val is not None:
                if op_release == "<":
                    if not (movie_release_val < rd_value):
                        continue
                elif op_release == "<=":
                    if not (movie_release_val <= rd_value):
                        continue
                elif op_release == ">":
                    if not (movie_release_val > rd_value):
                        continue
                elif op_release == ">=":
                    if not (movie_release_val >= rd_value):
                        continue
                elif op_release in ["=", "=="]:
                    if movie_release_val != rd_value:
                        continue

            if title not in unique_movies:
                unique_movies[title] = movie

        return list(unique_movies.values())[:3]

    def get_movie_poster(url):
        if not url or not url.startswith("http"):
            return None
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                image_data = base64.b64encode(response.content).decode('utf-8')
                return f"data:image/jpeg;base64,{image_data}"
        except requests.exceptions.RequestException:
            return None
        return None

    def generate_explanation(mood, user_input, movie):
        movie_description = (
            f"{movie.metadata.get('title', 'Unknown')} "
            f"({movie.metadata.get('genres', 'Unknown')})\n"
            f"Overview: {movie.metadata.get('overview', 'No overview available')}\n"
        )
        explanation_prompt = (
            f"A user is feeling {mood}. Based on their input, here is the recommended movie:\n"
            f"{movie_description}\n"
            "Generate a friendly, engaging explanation for why this movie is a great choice."
        )
        response = llm.invoke(explanation_prompt)
        return response.content if hasattr(response, "content") else response

    # -------------------------------------------
    # Streamlit UI
    # -------------------------------------------
    st.title("🎬 CineMood Chatbot: Discover Films That Match Your Mood! 🍿")

    # Add a subtitle and display the available genres in a formatted list.
    st.markdown("Welcome to CineMood Chatbot! 🤖")
    st.write("CineMood Chatbot will help you to get personalized movie recommendations based on your mood and preferences (i.e., genres, runtime, release date, production country, and production company).")
    st.markdown("To help you refine your movie search, here are some **sample moods** you might use:")
    st.markdown("- Happy, excited, grateful, hopeful, loving, proud, sad, lonely, hopeless, bored, guilty, tired, angry, frustrated, jealous, disgusted, anxious, fearful, overwhelmed, surprised, confused, indecisive, bittersweet, thoughtful, etc.")
    st.markdown("Here are **available genres** in my database:")
    st.markdown("- " + ", ".join(genres_list))
    st.write("Some sample queries you can try:")
    st.write("- I'm feeling happy and want to watch a joyful movie from genre music after 2020.")
    st.write("- I'm feeling bored and want to watch some movies before 2020 with runtime more than 120 minutes.")


    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_mood" not in st.session_state:
        st.session_state.user_mood = None
    if "user_genre" not in st.session_state:
        st.session_state.user_genre = None
    if "release_date" not in st.session_state:
        st.session_state.release_date = None
    if "cast" not in st.session_state:
        st.session_state.cast = None
    if "director" not in st.session_state:
        st.session_state.director = None
    if "production_country" not in st.session_state:
        st.session_state.production_country = None
    if "production_company" not in st.session_state:
        st.session_state.production_company = None
    if "runtime" not in st.session_state:
        st.session_state.runtime = None
    if "awaiting_restart_decision" not in st.session_state:
        st.session_state.awaiting_restart_decision = False
    if "awaiting_mood_clarification" not in st.session_state:
        st.session_state.awaiting_mood_clarification = False

    for message in st.session_state.messages:
        if message["type"] == "text":
            st.chat_message(message["role"]).write(message["content"])
        elif message["type"] == "image":
            st.chat_message(message["role"]).image(message["content"], width=200)

    user_input = st.chat_input("Tell me about your mood and any specific preferences...")

    if user_input:
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
        st.chat_message("user").write(user_input)

        if "fresh start" in user_input.lower():
            st.session_state.user_mood = "not specified"
            st.session_state.user_genre = "not specified"
            st.session_state.release_date = "not specified"
            st.session_state.cast = "not specified"
            st.session_state.director = "not specified"
            st.session_state.production_country = "not specified"
            st.session_state.production_company = "not specified"
            st.session_state.runtime = "not specified"

            response_text = "All preferences have been reset. Please provide your new mood and preferences."
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": response_text})
            st.chat_message("assistant").write(response_text)
            st.stop()

        conversation_generated = False

        if st.session_state.awaiting_mood_clarification:
            st.session_state.awaiting_mood_clarification = False
            if "keep current" in user_input.lower():
                new_detected_mood = st.session_state.user_mood
            else:
                candidate_mood = get_user_mood(user_input, valid_moods, genres_list)
                if candidate_mood:
                    new_detected_mood = candidate_mood
                    st.session_state.user_mood = candidate_mood
                else:
                    clarification_text = (
                        "I still couldn't detect a clear mood. "
                        "Please rephrase your mood or type 'keep current' to keep your current mood."
                    )
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": clarification_text})
                    st.chat_message("assistant").write(clarification_text)
                    st.session_state.awaiting_mood_clarification = True
                    st.stop()

            new_genre = detect_genre(user_input, valid_moods, genres_list)
            new_release_date = detect_release_date(user_input)
            new_cast = detect_cast(user_input)
            new_director = detect_director(user_input)
            new_production_country = detect_production_country(user_input)
            new_production_company = detect_production_company(user_input)
            new_runtime = detect_runtime(user_input)
            print(f"detected runtime is {new_runtime}")

        else:
            if not conversation_generated:
                short_reply = generate_conversational_response(user_input)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": short_reply})
                st.chat_message("assistant").write(short_reply)
                conversation_generated = True

            new_detected_mood = get_user_mood(user_input, valid_moods, genres_list)
            new_genre = detect_genre(user_input, valid_moods, genres_list)
            new_release_date = detect_release_date(user_input)
            new_cast = detect_cast(user_input)
            new_director = detect_director(user_input)
            new_production_country = detect_production_country(user_input)
            new_production_company = detect_production_company(user_input)
            new_runtime = detect_runtime(user_input)
            print(f"detected mood is {new_detected_mood}")
            print(f"detected genre is {new_genre}")
            print(f"detected release date is {new_release_date}")
            print(f"detected production country is {new_production_country}")
            print(f"detected runtime is {new_runtime}")

        if not new_detected_mood and st.session_state.user_mood not in [None, "not specified"]:
            clarification_text = (
                f"I couldn't detect a new mood. If you'd like to keep your current mood "
                f"({st.session_state.user_mood}) and your movie preferences, please type 'keep current' or provide details to modify your preferences. If you want a fresh start, please type 'fresh start' to reset all information, or provide details to modify your preferences."
            )
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": clarification_text})
            st.chat_message("assistant").write(clarification_text)
            st.session_state.awaiting_mood_clarification = True
            if new_genre:
                st.session_state.user_genre = new_genre
            if new_release_date:
                st.session_state.release_date = new_release_date
            if new_cast:
                st.session_state.cast = new_cast
            if new_director:
                st.session_state.director = new_director
            if new_production_country:
                st.session_state.production_country = new_production_country
            if new_production_company:
                st.session_state.production_company = new_production_company
            if new_runtime:
                st.session_state.runtime = new_runtime
            st.stop()
        elif new_detected_mood:
            st.session_state.user_mood = new_detected_mood
        else:
            if st.session_state.user_mood in [None, "not specified"]:
                clarification_text = "I couldn't detect your mood. Could you please state it clearly?"
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": clarification_text})
                st.chat_message("assistant").write(clarification_text)
                st.session_state.awaiting_mood_clarification = True
                st.stop()

        if new_genre:
            st.session_state.user_genre = new_genre
        if new_release_date:
            st.session_state.release_date = new_release_date
        if new_cast:
            st.session_state.cast = new_cast
        if new_director:
            st.session_state.director = new_director
        if new_production_country:
            st.session_state.production_country = new_production_country
        if new_production_company:
            st.session_state.production_company = new_production_company
        if new_runtime:
            st.session_state.runtime = new_runtime

        summary = (
            f"Detected mood: {st.session_state.user_mood}\n\n"
            f"Requested genre: {st.session_state.user_genre if st.session_state.user_genre else 'Not specified'}\n\n"
            f"Release date: {st.session_state.release_date if st.session_state.release_date else 'Not specified'}\n\n"
            f"Cast: {st.session_state.cast if st.session_state.cast else 'Not specified'}\n\n"
            f"Director: {st.session_state.director if st.session_state.director else 'Not specified'}\n\n"
            f"Production country: {st.session_state.production_country if st.session_state.production_country else 'Not specified'}\n\n"
            f"Production company: {st.session_state.production_company if st.session_state.production_company else 'Not specified'}\n\n"
            f"Runtime: {st.session_state.runtime if st.session_state.runtime else 'Not specified'}\n\n"
            "Here are the top movie choices based on your preferences:"
        )
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": summary})
        st.chat_message("assistant").write(summary)

        top_movies = get_movie_recommendations(
            mood=st.session_state.user_mood,
            genre=st.session_state.user_genre,
            release_date=st.session_state.release_date,
            cast=st.session_state.cast,
            director=st.session_state.director,
            production_country=st.session_state.production_country,
            production_company=st.session_state.production_company,
            runtime=st.session_state.runtime
        )

        if not top_movies:
            no_results_text = "Sorry, I couldn't find any movies matching your preferences. Try describing them differently!"
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": no_results_text})
            st.chat_message("assistant").write(no_results_text)
        else:
            for i, movie in enumerate(top_movies):
                metadata = movie.metadata
                movie_response = (
                    f"### {i+1}. {metadata.get('title', 'Unknown')} "
                    f"({metadata.get('genres', 'Unknown')})\n\n"
                    f"📅 **Release Date:** {metadata.get('release_date', 'Unknown')}\n\n"
                    f"🏷️ **Tagline:** {metadata.get('tagline', 'Unknown')}\n\n"
                    f"🎭 **Cast:** {metadata.get('main_cast', 'Unknown')}\n\n"
                    f"🎬 **Director:** {metadata.get('director', 'Unknown')}\n\n"
                    f"🌍 **Production Country:** {metadata.get('production_countries', 'Unknown')}\n\n"
                    f"🏢 **Production Company:** {metadata.get('production_companies', 'Unknown')}\n\n"
                    f"⏳ **Runtime:** {metadata.get('runtime', 'Unknown')} min\n\n"
                )
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": movie_response})
                st.chat_message("assistant").write(movie_response)
                
                poster_url = metadata.get('poster_path', '')
                if poster_url:
                    st.session_state.messages.append({"role": "assistant", "type": "image", "content": poster_url})
                    st.chat_message("assistant").image(poster_url, width=200)
                
                explanation = generate_explanation(st.session_state.user_mood, user_input, movie) + "\n\n"
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": explanation})
                st.chat_message("assistant").write(explanation)

        followup = (
            "Would you like to change any information or start with a fresh start? "
            "Type 'fresh start' to reset all information, or provide details to modify your preferences."
        )
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": followup})
        st.chat_message("assistant").write(followup)

    st.markdown("**Made by [Thanh Tung Vu](https://thanhtungvudata.github.io/)**")

if __name__ == "__main__":
    run_app()                
```

## **🎉 Results and Live Demo**

The final chatbot web app delivers movie recommendations based on the user's mood and preferences in just a second, with fresh content every week. 

You can try it here:👉 [CineMood Live App on Hugging Face Spaces](https://huggingface.co/spaces/thanhtungvudata/CineMood_Chatbot)

<div style="text-align: center;">
    <h3>🎬 Try CineMoodv3 Now!</h3>
    <iframe
        src="https://thanhtungvudata-cinemood-chatbot.hf.space"
        width="100%"
        height="600"
        style="border: none; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);"
        allowfullscreen>
    </iframe>
</div>

## **Tech Stack**
1. Data Storage & Retrieval
- Vector Database: ChromaDB (stores and retrieves movie embeddings efficiently for fast and scalable similarity search)
2. Embedding & RAG Pipeline
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

## 🎯 Conclusion & Next Steps

By integrating **conversational AI** with **retrieval-augmented generation (RAG)**, this chatbot offers **a truly personalized movie recommendation experience**. Key takeaways:

✅ **Conversational Interactions** make the system more **engaging and adaptive**.

✅ **Advanced Mood Detection** ensures **accurate recommendations** even from complex inputs.

✅ **Efficient Retrieval** enables **scalability** as the movie dataset grows.

### 🔜 What’s Next?

- Fine-tune the mood detection model** to better differentiate between similar moods.
- Expand the movie dataset** with real-time trending films from **TMDB API**.
- Enhance chatbot explanations** by adding **sentiment-aware storytelling**.
- Allow user feedback** to improve recommendation accuracy over time.


🚀 **Let me know what you think!**

---
📌 **Stay tuned for more updates on AI-powered movie recommendations!** 🎬✨

The code of this project is available [here](https://github.com/thanhtungvudata/CineMood_Chatbot). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
