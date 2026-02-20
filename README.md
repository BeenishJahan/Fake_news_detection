# 📰 Fake News Detection using Logistic Regression

## Overview

A binary classification model that predicts whether a news article is **Real** or **Fake**, using Logistic Regression and TF-IDF text vectorization.

---

## Dataset

**Source:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon on Kaggle.

~44,000 news articles split across two files — `Fake.csv` (unreliable sources, 2016–2017 US political news) and `True.csv` (verified Reuters articles). Each article contains a title, body text, subject, and date.

---

## Methodology

**1. Data Preparation** — Both files were loaded, labeled (Fake = 0, Real = 1), combined, and shuffled with a fixed random seed.

**2. Text Cleaning** — A custom function using Python's `re` library strips URLs, punctuation, numbers, and extra whitespace, and lowercases all text.

**3. TF-IDF Vectorization** — Text is converted to numerical vectors using TF-IDF, limited to the top 5,000 most meaningful words.

**4. Train/Test Split** — 70% training, 30% test.

**5. Model Training** — Logistic Regression trained on TF-IDF vectors.

**6. Evaluation** — Assessed using accuracy, confusion matrix, and classification report.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.43% |
| Precision (Fake) | 0.99 |
| Precision (Real) | 0.98 |
| Recall (Fake) | 0.98 |
| Recall (Real) | 0.99 |
| F1-Score (Both) | 0.98 |

212 mistakes out of 13,470 test articles.

---

## Limitations

- Trained on 2016–2017 US political news — may not generalize to other domains or time periods.
- The model learned writing style, not truth. A well-written fake article could fool it.
- TF-IDF has no understanding of context or meaning — more advanced models like BERT would perform better in production.
- Dramatic but real events can be misclassified as fake due to sensational language.

---

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Re

---

## How to Run

1. Clone the repository
2. Download `Fake.csv` and `True.csv` from the Kaggle link above and place them in the project folder
3. Open `fake_news_detection.ipynb` in VS Code or Jupyter
4. Run all cells in order
5. Use the `true_fake()` function at the end to test your own articles

---

## A Note from the Author

I'm a beginner in ML, currently working through Andrew Ng's Machine Learning Specialization on DeepLearning.AI. Logistic regression is the only classification algorithm I've covered so far. I built this project to apply what I'd learned to something I found genuinely fun rather than waiting until I knew more. It's not perfect, but it taught me more than any lesson could on its own.