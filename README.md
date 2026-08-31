# 🎬 Movie Recommendation System

A Machine Learning powered **Content-Based Movie Recommendation System** built with Python, Scikit-learn, TF-IDF, Cosine Similarity, OMDb API, and Streamlit.

The application recommends movies similar to a movie selected by the user and dynamically fetches movie details, IMDb ratings, genres, plots, and posters using the OMDb API.

---

## 🚀 Live Demo

🔗 **Live App:** 
https://movie-recommendation-system-ud43jlmvtaunmyoovijz9j.streamlit.app/

🔗 **GitHub:**  
https://github.com/Jay1912-pixel/Movie-Recommendation-System

---

## 📌 Project Overview

Finding a movie to watch can be difficult when thousands of movies are available.

This project solves that problem using a **Content-Based Recommendation System**.

The system analyzes movie-related textual information, converts it into numerical vectors using **TF-IDF**, and calculates similarity between movies using **Cosine Similarity**.

For example, if the user selects:

> **Inception**

the system identifies movies with similar content and recommends the most relevant results.

The application then uses the **OMDb API** to retrieve additional information such as:

- 🎬 Movie title
- 📅 Release year
- ⭐ IMDb rating
- 🎭 Genre
- 📝 Plot
- 🖼️ Movie poster

---

# 🎯 Objectives

The main objectives of this project are:

- Build a practical Machine Learning recommendation system.
- Apply NLP techniques to movie data.
- Convert textual movie information into numerical features.
- Implement TF-IDF vectorization.
- Calculate movie similarity using Cosine Similarity.
- Generate Top-N movie recommendations.
- Integrate a REST API for additional movie information.
- Build an interactive Streamlit application.
- Deploy the ML application for real-world usage.

---

# ✨ Features

### 🎬 Movie Recommendations

Select a movie and receive a list of similar movies.

### 🧠 Content-Based Filtering

Recommendations are generated based on movie content and textual features.

### 🔤 TF-IDF

Movie text is converted into numerical feature vectors using TF-IDF.

### 📐 Cosine Similarity

Similarity between movies is calculated using cosine similarity.

### ⭐ IMDb Ratings

IMDb ratings are retrieved dynamically through OMDb API.

### 🖼️ Movie Posters

Movie posters are displayed using API-provided image URLs.

### 📝 Movie Details

The application displays relevant movie information including:

- Title
- Year
- Rating
- Genre
- Plot
- Poster

### 🌐 API Integration

The project integrates the OMDb REST API to retrieve real-time movie metadata.

### 💻 Streamlit Interface

The ML model is exposed through an easy-to-use web interface.

---

# 🧠 Machine Learning Approach

## Content-Based Recommendation

This project uses **Content-Based Filtering**.

The basic idea is:

```text
Movie Content
     ↓
Text Processing
     ↓
TF-IDF Vectorization
     ↓
Numerical Movie Vectors
     ↓
Cosine Similarity
     ↓
Similarity Ranking
     ↓
Top-N Recommendations
