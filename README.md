# Sentiment Analysis

A sentiment analysis project that predicts whether a movie review is *positive* or *negative* based on its text. The model is built using Python with a small dataset of 1000 IMDB reviews and applies natural language processing (NLP) techniques along with machine learning.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Dataset](#dataset)
5. [Installation and Dependencies](#installation-and-dependencies)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Acknowledgments](#acknowledgments)

---

## Introduction

This project implements a sentiment analysis pipeline that:
1. Preprocesses text data (lowercasing, removing special characters, and stopwords).
2. Converts text into numerical features using *CountVectorizer*.
3. Trains a *Logistic Regression* model to classify reviews as positive or negative.
4. Evaluates the model using accuracy and a classification report.

---

## Project Structure


📂 Sentiment-Analysis
├── sentiment_analysis.ipynb   # Jupyter Notebook with code and execution
├── README.md                  # Documentation for the project
├── requirements.txt           # List of dependencies
└── IMDB Dataset.csv           # Dataset file (not included, see instructions below)


---

## Features

- Text preprocessing (lowercase, special characters removal, and stopwords removal).
- Feature extraction using *CountVectorizer*.
- Sentiment classification using *Logistic Regression*.
- Performance evaluation using accuracy, precision, recall, and F1-score.

---

## Dataset

- *Dataset Name*: IMDB Movie Reviews
- *Source*: [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- *Size*: A small subset of 1000 reviews (positive and negative).

Download the dataset and place the file IMDB Dataset.csv in the root directory of the project.

---

## Installation and Dependencies

1. Clone the repository:
   bash
   git clone https://github.com/your-username/Sentiment-Analysis.git
   cd Sentiment-Analysis
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Required libraries:
   - pandas
   - numpy
   - scikit-learn
   - nltk

4. Download NLTK stopwords:
   Run this code in your notebook or Python environment:
   python
   import nltk
   nltk.download('stopwords')
   

---

## How to Run

1. *Start Jupyter Notebook*:
   bash
   jupyter notebook
   

2. Open the sentiment_analysis.ipynb file.

3. Execute each cell step by step to:
   - Load and preprocess the dataset.
   - Train the Logistic Regression model.
   - Evaluate the model.
   - Predict sentiment for custom reviews.

---

## Results

- *Model Accuracy*: ~88% (can vary depending on the dataset and preprocessing).
- Example Sentiment Prediction:
  
  Review: "This movie was absolutely amazing! I loved it."
  Sentiment: Positive
  

---

## Acknowledgments

- Dataset by [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
- Inspired by basic NLP and machine learning workflows.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
