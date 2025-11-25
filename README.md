Email Spam Classifier using NLP and Machine Learning
Project Overview

This project is an Email Spam Detection System that classifies emails into spam and ham (non-spam) using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system helps in automatically filtering unwanted emails and improving email productivity.

Dataset

Enron Email Spam Dataset from HuggingFace (bvk/ENRON-spam).

Contains around 35,000 real emails.

Each email includes:

Full email content (subject + body)

Label: 0 for ham, 1 for spam

Filename (source reference)

Technologies Used

Python 3 and Google Colab

NLTK for text preprocessing: tokenization, stopwords removal, lemmatization

Scikit-learn for machine learning: Naive Bayes classifier, TF-IDF vectorization, evaluation

Joblib for saving and loading models

Project Workflow

Data Loading: Extracted emails and labels from the dataset.

Text Preprocessing:

Cleaned text by removing URLs, special characters, and whitespace

Tokenization, stopwords removal, and lemmatization

Feature Extraction: Converted text to numerical features using TF-IDF vectorization.

Train/Test Split: Divided dataset for training and evaluation.

Model Training: Used Multinomial Naive Bayes for spam classification.

Model Evaluation: Evaluated with accuracy, precision, recall, and F1-score.

Prediction Function: Built a function for classifying new emails as spam or ham.

Model Saving: Saved the trained model and vectorizer for future use.

Challenges

Handling nested email lists and dataset column variations

NLTK resource installation issues in Colab

Preprocessing textual data effectively to improve model performance

Results

Successfully classified emails into spam and ham with high accuracy (~95%+).

Preprocessing combined with TF-IDF and Naive Bayes yielded reliable results.

The model can be used for real-time email spam filtering.
