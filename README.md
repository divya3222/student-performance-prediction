# Student Performance Prediction Using Machine Learning

## Project Overview

This project develops a Machine Learning system for predicting student academic performance. The system analyzes student academic, demographic, family, and social factors and predicts the student's final academic performance.

The project was developed as part of the Data Science Master Virtual Internship at EduSkills.

## Objective

The main objectives of this project are:

- Analyze student academic data
- Perform data preprocessing and cleaning
- Explore relationships between student attributes and performance
- Build Machine Learning models
- Predict final student performance
- Identify students who may be academically at risk
- Compare different Machine Learning algorithms

## Dataset

The dataset contains student-related academic and personal attributes.

Important features include:

- `G1` – First-period grade
- `G2` – Second-period grade
- `G3` – Final grade
- `studytime` – Weekly study time
- `absences` – Number of school absences
- `failures` – Number of previous failures
- `age` – Student age
- `schoolsup` – Extra educational support
- `famsup` – Family educational support
- `higher` – Intention to pursue higher education
- `health` – Health status
- `freetime` – Free time after school
- `goout` – Going out with friends

The target variable for the prediction task is `G3`, the final student grade.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Google Colab

## Machine Learning Models

### Regression Models

The following models are used to predict the final grade:

- Linear Regression
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Random Forest Regressor

### Classification Models

Students are also classified into academic-risk categories using:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Random Forest Classifier

## Project Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Selection
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Student Performance Prediction
