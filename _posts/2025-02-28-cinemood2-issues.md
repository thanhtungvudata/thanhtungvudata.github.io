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
