# Milk Quality Prediction

This project implements a machine learning pipeline to predict the quality grade of milk based on various physicochemical properties. Multiple classification algorithms are applied, tuned, and evaluated to find the best model for accurate milk quality classification.

---

## Dataset

The dataset used in this project is sourced from [Kaggle - Milk Quality Dataset](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). It contains physicochemical properties of milk samples along with a quality grade label.

---

## Project Overview

The goal of this project is to build and evaluate various machine learning models to predict the milk grade based on the features provided in the dataset.

### Key steps:

- Data loading and exploration
- Data preprocessing including label encoding
- Train-test split
- Training and hyperparameter tuning of several classifiers
- Model evaluation using accuracy, precision, recall, and confusion matrices
- Ensemble learning using stacking classifier

---

## Technologies & Libraries Used

- Python 3.x
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn (visualization)
- Scikit-learn (machine learning algorithms, metrics, and model tuning)
- mlxtend (stacking ensemble classifier)

---

## Models Implemented

- Naive Bayes (GaussianNB)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Multi-layer Perceptron (MLP) Classifier
- Ensemble Learning using StackingCVClassifier

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
