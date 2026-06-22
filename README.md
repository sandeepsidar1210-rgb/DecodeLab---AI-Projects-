# 🚀 DecodeLabs - Artificial Intelligence Industrial Training

**Author:** Sandeep Kumar Sidar  
**Institution:** International Institute of Information Technology (IIIT), Naya Raipur  
**Domain:** Artificial Intelligence  

---

## 📌 Overview
This repository contains the complete progression of the **Artificial Intelligence Industrial Training** powered by DecodeLabs. The journey is strategically divided into three major milestones, starting from foundational deterministic logic, moving to supervised machine learning, and culminating in a commercial-grade active prediction engine.

---

## 📂 Project 1: Rule-Based Chatbot (LogicBot)
**Focus:** Conversational AI & Deterministic Logic  
The training commenced with building a foundational conversational agent named **LogicBot**. This phase focused on understanding how machines process text and execute actions using structured dictionary lookups.

* **Objective:** To engineer an interactive chatbot capable of parsing user inputs and responding based on predefined string matching and fallback rules.
* **Key Implementations:** * **Input Sanitization:** Normalizing user input using `.lower().strip()` for reliable matching.
  * **Dictionary Routing:** Using Python dictionaries (`responses.get`) for O(1) time complexity intent matching.
  * **Graceful Fallbacks:** Handling out-of-scope queries with predefined default responses.
  * **Continuous Loop:** Implementing an infinite `while` loop with specific exit conditions (`exit`, `quit`).
* **Tech Stack:** Pure Python.

---

## 📂 Project 2: KNN Classification Engine
**Focus:** Supervised Machine Learning & Data Preprocessing  
Moving from hardcoded logic to data-driven pattern recognition, this project focused on categorizing data using the K-Nearest Neighbors algorithm on the classic Iris dataset.

* **Objective:** To build a robust classification model capable of accurately predicting biological classes (Setosa, Versicolor, Virginica) based on physical measurements.
* **Key Implementations:**
  * **Data Splitting:** 80/20 train-test split using `train_test_split` with a fixed `random_state` for reproducibility.
  * **Feature Scaling:** Implementing `StandardScaler` to normalize feature variance, crucially fitting only on training data to prevent data leakage.
  * **Model Training:** Utilizing `KNeighborsClassifier` with an optimal $K=5$ (Elbow Point) to prevent overfitting/underfitting.
  * **Validation & Metrics:** Evaluating the model using Accuracy, weighted F1 Score, Confusion Matrices, and detailed classification reports.
  * **Live Inference:** Built a custom input pipeline for real-time user predictions and probability scoring (`predict_proba`).
* **Tech Stack:** Python, Scikit-Learn.

---

## 📂 Project 3: The Digital Matchmaker (Tech Stack Recommender)
**Focus:** Active Prediction & Content-Based Filtering  
This capstone project marks a fundamental shift in algorithmic intent—solving "Choice Overload" by building an engine that understands user intent with absolute precision. 

* **Objective:** To engineer a **Tech Stack Recommender** that maps a user's raw skills to specific tools and job roles based on a dataset (`raw_skills.csv`).
* **Core Logic & Architecture:**
  * **TF-IDF Vectorization:** Upgraded feature extraction from simple binary overlap to Term Frequency-Inverse Document Frequency (TF-IDF). This penalizes generic words and rewards highly specific technical tags.
  * **Cosine Similarity:** Measured the mathematical angle between the user's input vector and job role vectors to find semantic alignment regardless of vector magnitude.
  * **Top-N Pipeline:** Automated the sorting of similarity scores and filtered the output to provide the Top 3 recommendations.
  * **Skill Gap Analysis:** Utilized Python `set` intersections and differences to dynamically calculate and display overlapping skills and top missing skills for career growth.
* **Outcome:** A visually rich, terminal-based production-grade recommendation logic skeleton capable of flawless matches.
* **Tech Stack:** Python, Pandas, NumPy, Scikit-Learn.

---

## 🛠️ Core Technologies Used
* **Programming:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (KNN, StandardScaler, TF-IDF Vectorizer, Cosine Similarity)
* **Environment:** Terminal / CLI

---
*Developed as part of the DecodeLabs 2026 Batch AI Track.*
