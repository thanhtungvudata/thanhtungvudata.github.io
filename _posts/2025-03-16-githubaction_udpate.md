---
title: "Automating Database Updates with GitHub Actions"
date: 2025-03-16
image: /assets/images/cinemoodchatbot_overview.png
categories:
  - Data Science Projects
tags:
  - Data Science
  - AI Agent
  - ML Engineer
  - RAG
---

# Automating Database Updates with GitHub Actions

## Introduction
In many applications, keeping the database up-to-date with fresh data from an API is essential. Manually updating data can be time-consuming and error-prone. GitHub Actions provides a robust way to automate these updates on a schedule.

In this blog post, we'll explore how to use GitHub Actions to schedule and automate database updates from an API. We'll then illustrate this with a real-world example: updating the database for a **movie recommendation chatbot** that suggests trending movies based on a user's mood and preferences.

## Why Automate Database Updates?
Automating database updates using GitHub Actions has several advantages:

- **Ensures fresh data**: The latest information is always available.
- **Eliminates manual work**: Reduces the risk of human error.
- **Improves efficiency**: Runs automatically without requiring intervention.
- **Keeps repositories in sync**: Updates and stores new data for continuous deployment.

## Setting Up a GitHub Actions Workflow
To automate database updates from an API, follow these steps:

### 1️⃣ Define the Workflow File
Create a file in your GitHub repository under `.github/workflows/update_db.yml`.

```yaml
name: Auto-Update Database

on:
  schedule:
    - cron: "0 0 * * 1"  # Runs every Monday at midnight UTC
  workflow_dispatch:  # Allows manual trigger from GitHub Actions

jobs:
  update-database:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install Dependencies
      run: pip install -r requirements.txt

    - name: Fetch New Data from API
      env:
        API_KEY: ${{ secrets.API_KEY }}
      run: python fetch_data.py

    - name: Process and Store Data
      run: python process_data.py

    - name: Commit and Push Updates
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        git config --global user.email "github-actions[bot]@users.noreply.github.com"
        git config --global user.name "GitHub Actions"
        git add data/
        git commit -m "Auto-update database [skip ci]" || echo "No changes to commit"
        git push https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/YOUR_USERNAME/YOUR_REPO.git HEAD:main
```

### 2️⃣ Explanation of Workflow Steps
- **Schedule Execution**: The job runs every Monday at midnight UTC (for an example).
- **Manual Execution**: You can also trigger it manually from GitHub Actions.
- **Checkout Repository**: Ensures the latest code is available.
- **Set Up Python**: Installs Python for script execution.
- **Install Dependencies**: Installs required Python libraries.
- **Fetch Data**: Calls an API to retrieve new data.
- **Process Data**: Cleans, transforms, and stores the fetched data.
- **Commit & Push Updates**: Stores the new data in the repository.

## Example: Updating a Movie Recommendation Chatbot
Now, let's apply this setup to a **chatbot** that suggests trending movies based on user moods and preferences. The workflow will:

1. Fetch **trending movies** from an API.
2. Generate **movie embeddings** using OpenAI.
3. Store embeddings in a **ChromaDB vector database**.
4. Push updates to **GitHub and deploy to Hugging Face Spaces**.

### 🔹 Workflow for the Movie Chatbot
```yaml
name: Auto-Update Vector Database Weekly & Deploy to Hugging Face

on:
  schedule:
    - cron: "0 0 * * 1"  # Runs every Monday at midnight UTC
  workflow_dispatch:  # Allows manual trigger from GitHub Actions

jobs:
  update-database:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install Dependencies
      run: pip install -r requirements.txt

    - name: Remove Old ChromaDB Before Storing New Data
      run: |
        rm -rf chroma_db/  # Remove old ChromaDB files
        mkdir -p chroma_db  # Ensure the directory is recreated

    - name: Fetch Trending Movies
      env:
        TMDB_API_KEY: ${{ secrets.TMDB_API_KEY }}
      run: python fetch_movies.py

    - name: Generate Movie Embeddings
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: python generate_embeddings.py

    - name: Store Embeddings in ChromaDB
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: python store_in_chromadb.py

    - name: Commit and Push Updates to GitHub
      run: |
        git config --global user.email "actions@github.com"
        git config --global user.name "GitHub Actions"
        git add trending_movies.json movie_embeddings.json chroma_db/
        git commit -m "Auto-update movies database [skip ci]" || echo "No changes to commit"
        git push origin main

  deploy-to-huggingface:
    needs: update-database
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Install and Configure Git LFS
      run: |
        sudo apt-get update
        sudo apt-get install -y git-lfs
        git lfs install

    - name: Clone Hugging Face Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        git config --global credential.helper store
        echo "https://user:$HF_TOKEN@huggingface.co" > ~/.git-credentials
        git config --global credential.useHttpPath true
        git config --global user.email "github-actions[bot]@users.noreply.github.com"
        git config --global user.name "GitHub Actions"
        GIT_LFS_SKIP_SMUDGE=1 git clone https://user:$HF_TOKEN@huggingface.co/spaces/thanhtungvudata/CineMood_Chatbot CineMood_Chatbot
        cd CineMood_Chatbot
        git lfs pull

    - name: Remove Old ChromaDB and Movie Files in Hugging Face Space
      run: |
        cd CineMood_Chatbot
        rm -rf chroma_db/  # ✅ Remove old ChromaDB to prevent duplicates
        rm -f trending_movies.json movie_embeddings.json  # Remove old movie files

    - name: Copy Updated Data to Hugging Face Repo
      run: |
        mkdir -p CineMood_Chatbot/chroma_db  # Ensure target directory exists
        rsync -av --ignore-missing-args --exclude='.git' ./chroma_db/ CineMood_Chatbot/chroma_db/
        rsync -av --exclude='.git' trending_movies.json movie_embeddings.json CineMood_Chatbot/

    - name: Configure Git LFS for Large Files
      run: |
        cd CineMood_Chatbot
        git lfs track "chroma_db/chroma.sqlite3"
        git add .gitattributes

    - name: Commit and Push Updates to Hugging Face
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        cd CineMood_Chatbot
        git add chroma_db/ trending_movies.json movie_embeddings.json
        git commit -m "Auto-update from GitHub" || echo "No changes to commit"
        git push https://user:$HF_TOKEN@huggingface.co/spaces/thanhtungvudata/CineMood_Chatbot.git HEAD:main
```

### 🔹 Explanation of the Chatbot Workflow
- **Fetch Trending Movies**: Retrieves current trending movies from TMDb.
- **Generate Movie Embeddings**: Uses OpenAI to convert movie descriptions into vector embeddings.
- **Store Embeddings in ChromaDB**: Saves the embeddings in a vector database.
- **Commit & Push Updates**: Updates GitHub with the new database.
-  **Deploy Updates**: updates the new database to Hugging Face Spaces for deployment

## Running the Workflow
1. **Automatic Execution**: The job runs weekly based on the cron schedule.
2. **Manual Trigger**: You can run it manually from the GitHub Actions tab.
3. **Logs & Debugging**: Check logs in GitHub Actions to debug errors.

## Conclusion
With GitHub Actions, automating database updates is simple and efficient. Whether you’re fetching trending movies, weather data, or stock prices, scheduled workflows ensure your application stays updated with minimal manual intervention.

---
🚀 Start automating your database updates today and let GitHub Actions handle the rest!

The code of an example project using GitHub Action is available [here](https://github.com/thanhtungvudata/CineMood_Chatbot). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
