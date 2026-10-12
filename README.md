# 🎬 Netflix Movies & TV Shows Clustering & Recommender

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Unsupervised](https://img.shields.io/badge/Machine%20Learning-Unsupervised%20Clustering-brightgreen?style=for-the-badge)](https://scikit-learn.org)
[![K-Means](https://img.shields.io/badge/Clustering-K--Means%20%26%20Hierarchical-blue?style=for-the-badge)](https://scikit-learn.org)
[![PCA](https://img.shields.io/badge/Dimensionality%20Reduction-PCA%20(4k%20components)-orange?style=for-the-badge)](https://scikit-learn.org)
[![Recommender](https://img.shields.io/badge/Engine-Content--Based%20Recommender-red?style=for-the-badge)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An unsupervised learning and Natural Language Processing project clustering 7,787 Netflix movies and TV shows into thematic cohorts based on metadata, director networks, genres, and synopsis descriptions. Includes Principal Component Analysis (PCA) dimensionality reduction and a content-based recommendation system powered by cosine similarity.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Dataset & Exploratory Analysis](#-dataset--exploratory-analysis)
- [NLP & Dimensionality Reduction (PCA)](#-nlp--dimensionality-reduction-pca)
- [Clustering Methodologies](#-clustering-methodologies)
- [Content-Based Recommender System](#-content-based-recommender-system)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Author & Connect](#-author--connect)

---

## 🚀 Executive Overview
With thousands of titles in the global Netflix library, effective content grouping prevents catalog fragmentation and powers discovery algorithms.

This project processes **7,787 titles and 11 metadata attributes** through an unsupervised NLP pipeline:
- **Corpus Tokenization:** Merges cast, directors, producing countries, genres, maturity ratings, and synopses into a 10,000-dimensional TF-IDF vector space.
- **PCA Dimensionality Reduction:** Reduces dimensionality to 4,000 components while capturing >80% of cumulative variance.
- **Dual Clustering:** Determines optimal cluster counts using the Elbow Method and Silhouette Analysis (K-Means: $k=6$; Agglomerative Hierarchical: $k=7$).
- **Content-Based Recommendations:** Generates top-10 personalized title recommendations via cosine similarity matrices.

---

## 🏗️ System Architecture

```text
[ Netflix Media Library: 7,787 Titles (NETFLIX_MOVIES.csv) ]
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│             Exploratory Analysis & Wrangling           │
│   • Movies (~70%) vs TV Shows (~30%) distribution      │
│   • Imputation of missing directors, cast & country    │
│   • Release year trends & runtime parsing              │
└────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│           Feature Fusion & TF-IDF Vectorization        │
│   • Merged text attributes: Director + Cast + Country  │
│     + Listed_in + Rating + Description                 │
│   • N-Gram TF-IDF Vectorizer (10,000 features)         │
└────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│             Principal Component Analysis (PCA)         │
│   • Retains 4,000 orthogonal components                │
│   • Preserves >80% total explained variance            │
└────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┴──────────────────────┐
       ▼                                             ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│    K-Means Clustering       │       │ Agglomerative Hierarchical  │
│  Optimal k=6 (Silhouette)   │       │  Optimal k=7 (Dendrogram)   │
└─────────────────────────────┘       └─────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│         Content-Based Recommendation Engine            │
│   Computes Cosine Similarity on Title Feature Vectors  │
│   Returns Top 10 Closest Semantic Matches              │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Exploratory Analysis
- **Observations:** 7,787 movies and TV shows.
- **Key Findings:**
  - Movies dramatically outnumber TV shows (approx. 70% vs. 30%), though TV show additions grew exponentially post-2016.
  - The United States is the top producing country, followed by India and the United Kingdom.
  - Documentaries, Stand-Up Comedy, and International Dramas represent the fastest-growing genres.

---

## 🧠 Clustering Methodologies
1. **K-Means Clustering:** Evaluated inertia curves using the Elbow Method; optimal cluster separation achieved at **$k=6$** with positive Silhouette scores.
2. **Agglomerative Hierarchical Clustering:** Visualized complete-linkage dendrograms; identified **$k=7$** natural thematic clusters (e.g., Bollywood family dramas, British crime thrillers, Anime & animated series).

---

## 🎯 Content-Based Recommender System
```python
# Recommender Function Preview
def recommend_netflix_titles(title, cosine_sim_matrix, top_n=10):
    idx = title_indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [titles[i[0]] for i in sim_scores]
```

---

## 📁 Repository Structure
```text
├── Netflix_movie_show.ipynb  # Comprehensive EDA, TF-IDF, PCA, clustering & recommender notebook
├── NETFLIX_MOVIES.csv        # Netflix catalog dataset
├── Presentation.pptx         # Executive project presentation deck
└── README.md                 # Detailed project documentation
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/abhik99/Netflix_movies_tvshows.git
cd Netflix_movies_tvshows
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Launch Notebook
```bash
jupyter notebook Netflix_movie_show.ipynb
```

---

## 👨‍💻 Author & Connect

**Abhishek Kumar (Abhi)**  
*Data Scientist & Analytics Engineer*

- 🌐 **Portfolio:** [abhicodes.online](https://abhicodes.online)
- 💼 **LinkedIn:** [linkedin.com/in/abhik99](https://linkedin.com/in/abhik99)
- 🐙 **GitHub:** [@abhik99](https://github.com/abhik99)
- 📧 **Email:** [akdevil27boy@gmail.com](mailto:akdevil27boy@gmail.com)

---
*Distributed under the MIT License.*
