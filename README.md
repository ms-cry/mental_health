# Mental Health Sentiment Detection Using YouTube Comments

## Project Overview

This project is an AI-based Mental Health Sentiment Detection System that analyzes YouTube comments and classifies them into emotional categories using Machine Learning and Natural Language Processing (NLP).

The system fetches YouTube comments using the YouTube Data API, preprocesses the text, converts it into numerical features using TF-IDF, and uses Logistic Regression to classify comments into:

* Normal
* Sad
* Anxiety
* Depressive Indicators

The goal of this project is to provide a supportive emotional analysis tool and not a medical diagnosis system.

---

## Features

* Fetch YouTube comments using video URL
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression model training
* Prediction of emotional categories
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Safe and responsible output generation

---

## Project Structure

```bash
mental-health-sentiment-app/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── youtube/
│
├── models/
│   ├── logistic_model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── config/
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── youtube_fetcher.py
│   │   └── data_loader.py
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py
│   │
│   ├── features/
│   │   └── tfidf_vectorizer.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── utils/
│       └── helpers.py
│
├── app/
│   └── app.py
│
├── notebooks/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

### Programming Language

* Python 3.10+

### Libraries

* pandas
* numpy
* scikit-learn
* joblib
* matplotlib
* google-api-python-client
* streamlit (optional)

### ML Techniques

* TF-IDF Vectorization
* Logistic Regression

---

## Installation

### Step 1: Clone Repository

```bash
git clone <your-github-repo-link>
cd mental-health-sentiment-app
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

### Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

---

## Dataset

Use a Kaggle dataset related to mental health sentiment classification.

Recommended labels:

* normal
* sad
* anxiety
* depressive indicators

Place dataset here:

```bash
data/raw/dataset.csv
```

Required columns:

```csv
text,label
"I feel tired all the time",3
"This made me happy today",0
```

---

## YouTube API Setup

Create API credentials from Google Cloud and enable:

## YouTube Data API v3

Add your API key inside:

```python
src/config/config.py
```

Example:

```python
YOUTUBE_API_KEY = "your_api_key_here"
```

---

## Model Training

Run:

```bash
python -m src.models.train
```

This will:

* load dataset
* preprocess text
* apply TF-IDF
* train Logistic Regression
* evaluate performance
* save model and vectorizer

Saved files:

```bash
models/logistic_model.pkl
models/vectorizer.pkl
```

---

## Prediction

Run:

```bash
python -m src.models.predict
```

This will:

* load saved model
* fetch YouTube comments
* preprocess comments
* predict emotional labels

---

## Sample Output

### Input Comment

```text
I feel so lost and empty these days
```

### Predicted Label

```text
Depressive Indicators
```

### Final Output

```text
This comment shows strong negative emotional signals and may reflect emotional distress.
```

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score (Macro)
* F1-score (Weighted)
* Confusion Matrix

F1-score is prioritized because the dataset may contain class imbalance.

---

## Important Note

This project is NOT a medical diagnosis system.

It is only an AI-based emotional analysis tool for educational and research purposes.

Do not use it for clinical decision-making.

---

## Future Improvements

* Upgrade to BERT / RoBERTa
* Add Streamlit dashboard
* Real-time sentiment monitoring
* Mood tracking visualization
* Multi-language support
* Better class balancing

---

## Author

Project developed for academic submission in Machine Learning / AI-based sentiment analysis.

---
