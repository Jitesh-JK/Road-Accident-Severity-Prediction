
# Road Accident Severity Prediction

## Project Overview

Road Accident Severity Prediction is a Machine Learning project that aims to predict the severity of road accidents based on various accident-related features.

The project uses historical road accident data to train and evaluate multiple Machine Learning classification algorithms. A Streamlit web application is also developed to allow users to enter accident details and obtain a predicted severity category.

## Problem Statement

Road accidents are a major concern in India. Understanding the factors associated with accident severity can help in analysing accident patterns and supporting road safety awareness.

This project focuses on developing a Machine Learning-based system that predicts accident severity using available accident-related information.

## Objectives

- Analyse historical road accident data.
- Perform data cleaning and preprocessing.
- Explore accident patterns through data visualisation.
- Implement and compare multiple Machine Learning algorithms.
- Evaluate models using suitable performance metrics.
- Develop a web application for accident severity prediction.

## Dataset

- **Dataset file:** `ML_DataSet.csv`
- **Number of records:** 4,053
- **Number of columns:** 12

The dataset contains accident-related information used for data analysis and model development.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Workflow

1. Data Loading and Understanding
2. Data Cleaning
3. Data Processing
4. Exploratory Data Analysis (EDA)
5. Model Training
6. Model Evaluation and Comparison
7. Final Model Selection
8. Model Deployment using Streamlit

## Machine Learning Algorithms

The project implements and compares the following algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Balanced Random Forest (final selected model)

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

Detailed experimental results, visualisations and model comparisons are available in the `main.ipynb` notebook.

## Accident Severity Categories

The application predicts one of the following categories:

1. Fatal
2. Grievous Injury
3. Minor Injury
4. No Injury

## Project Structure

```text
Road-Accident-Severity-Prediction/
│
├── app.py
├── main.ipynb
├── ML_DataSet.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── artifacts/
│   └── road_accident_model.joblib
│
└── assets/
    └── accident.jpg
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Road-Accident-Severity-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Road-Accident-Severity-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Application Features

- User-friendly interface.
- Accident-related information input.
- Machine Learning-based severity prediction.
- Model performance comparison.
- Prediction result display.

## Conclusion

This project demonstrates the application of Machine Learning techniques to road accident severity prediction. It includes data preprocessing, exploratory analysis, model training, performance evaluation and a working web-based prototype.

The project provides an interactive way to explore how accident-related features can be used to predict accident severity.
