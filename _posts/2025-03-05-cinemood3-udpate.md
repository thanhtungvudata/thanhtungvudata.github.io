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
In my previous [post](https://thanhtungvudata.github.io/data%20science%20projects/cinemood2-issues/), I built a **mood-based movie recommendation app** that used **LLM (Large Language Model)** to scan through around **50 trending movie metadata** and select the **top 3 movies** based on user mood. The app handled **validation and hallucination** in a simple way by letting LLM chec the user input with a list of valid moods. However, the method had limitations in working with **larger datasets**, ensuring **reliable validation**, and **reducing LLM hallucination.**

<img src="/assets/images/cinemood2_overview2.png" alt="CineMood2" width="600">

In this post, I take a step further by implementing a **RAG (Retrieval-Augmented Generation) mood-based trending movie recommendation app** that can efficiently **handle a larger dataset of movies**, validate moods more effectively using **embeddings and similarity scores**, and improve the **explainability of recommendations**.

## **Limitations of the Previous Approach**
While the previous approach provided **decent recommendations**, it had several shortcomings:

1. **Scalability Issues**: LLM-based approaches require processing a predefined, **limited set of movies** (e.g., 50 movies) at inference time. Increasing the dataset size exponentially increases computation time and memory usage, leading to impractical delays.
2. **Validation of Mood**: The mood validation was **not robust**—it relied only on simple keyword matching, which fails to capture semantic similarity. This means they might misinterpret user moods, leading to **incorrect recommendations**.  
3. **LLM Hallucination**: Since the model relied solely on **LLM reasoning** to select movies, it could **hallucinate recommendations** not present in the dataset. This is because LLMs, when not grounded in structured data, **tend to hallucinate**—generating movie titles, summaries, or recommendations that do not exist because their responses are based purely on learned probabilities rather than factual data.


To overcome these limitations, I implemented **RAG (Retrieval-Augmented Generation)**, which enhances the accuracy and explainability of recommendations.

## **What is RAG?**
**Retrieval-Augmented Generation (RAG)** is a method that combines **information retrieval (IR) and generative AI** to improve text generation by grounding responses in **real-world data**. Instead of relying **solely on the LLM’s internal knowledge**, RAG retrieves **relevant documents** or data points from an external **vector database** before generating responses.

## **RAG-Based Workflow in My App**
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


## **Results & Improvements**
With this RAG-based approach, the **movie recommendations are more accurate, scalable, and explainable**. The app now:
- ✅ Works with **large-scale datasets** stored in ChromaDB.
- ✅ Validates moods **more effectively** using **vector similarity**.
- ✅ Reduces **hallucination by grounding explanations in real movie metadata**.
- ✅ Provides **dynamic explanations using LLM** while maintaining factual correctness.

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
