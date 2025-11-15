# Embedding-CosineSimilarity-Classifier-AdulData
Embedding&amp;CosineSimilarity and Classifier Functions + Adult Data overview

# Gender Bias in Text Embeddings & Census Income Models

This repository contains three complete, reproducible pipelines for analyzing bias in machine learning models:

1. Embedding Bias Pipeline — generates OpenAI embeddings and evaluates gender & sentiment bias.
2. Bias Curve Pipeline — measures how classifier bias changes with different sampling distributions.
3. Adult Dataset (UCI Census Income) Analysis — studies gender bias and debiasing on real economic data.

All code is modular, documented, and safe for public use (no API keys stored).

---

# Features

## Embedding Pipeline
- Uses OpenAI `text-embedding-3-large`
- PCA embedding visualization
- SVC sentiment & gender classifiers
- Cosine-similarity bias metric
- 1000-sample permutation significance test
- Saves embeddings to CSV and pickle

## Bias Curve Pipeline
- Tests SVC, Logistic Regression, Random Forest, Gradient Boosting, MLP, Linear Regression
- Computes numeric bias across training sampling distributions (0 → 100%)
- Produces interactive Plotly bias curves
- Supports flexible model kwargs

## Adult Dataset (Census Income) Analysis
- Cleans UCI Adult dataset
- Logistic regression for income prediction
- Measures prediction gap by gender
- Shows 6-hour weekly work gap between men vs women
- Debiases model by adjusting working hours
- Explains how structural inequality creates model bias


---

# Installation

## 1. Clone the repo

## 2. Install Python dependencies

## 3. Set your OpenAI API key (for embedding generation)

### macOS / Linux

---

# Run the Embedding Pipeline

Runs PCA, classifiers, cosine similarity, permutation test, and prints results.


---

# Run the Bias Curve Pipeline

Produces a DataFrame and interactive Plotly figure.

Example:


---

# Adult Dataset (UCI Census Income) Analysis

This reproduces the gender bias experiment from your Colab notebook.

## 1. Download the dataset
Download from UCI: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data


Save as: "adult.data"


---

## 2. Clean and preprocess


---

## 3. Train logistic regression & measure gender bias


---

## 4. Measure work-hour imbalances (structural inequality)

Women work ~6 fewer hours/week** → model encodes this inequality.

---

## 5. Debiasing by correcting hours worked


This significantly reduces the prediction gap.

---

# Interpretation

These pipelines demonstrate:

- Text embeddings encode gender-sensitive information
- Sentiment and gender directions correlate
- Classifiers show different bias levels depending on sampling
- Real-world datasets contain structural inequalities
- Fairness interventions (e.g., correcting hours worked) reduce bias

---

# Contributions

Pull requests and issues are welcome.

---

# Contact

For questions or collaboration, contact via GitHub.





