# efiko

## Dataset Description:
The aim of the dataset is to understand the influence of various factors like economic, personal and social on a student's performance.

## Dataset Features:
  - gender
  - race/ethnicity
  - parental level of education
  - lunch
  - test preparation course
  - math score
  - reading score
  - writing score

## Target Feature: `math score`

## Dataset Source Link : https://www.kaggle.com/code/spscientist/student-performance-in-exams/notebook

## Project Objective:
The objective of this project is to predict the math score of a student based on the given features.

## Project Tasks:

### Data Ingestion

- Read the data as a CSV file.
- Split the data into training and testing sets.
- Save the training and testing sets as CSV files.

### Data Transformation

- Create a ColumnTransformer pipeline.
- Apply SimpleImputer with median strategy for numeric variables.
- Scale numeric data with StandardScaler.
- Apply SimpleImputer with most frequent strategy for categorical variables.
- Perform ordinal encoding on categorical variables.
- Scale categorical data with StandardScaler.
- Save the preprocessor as a pickle file.

### Model Training

- Evaluate the performance of base models.
- Identify the best performing model (CatBoostRegressor).
- Perform hyperparameter tuning
- Save the result as a pickle file.

### Prediction Pipeline

- Convert input data into a DataFrame.
- Load pickle files for data preprocessing and model prediction.
- Predict final results.

### API Creation

- Develop a Flask application with a user interface for predicting the students math score in a web application.

## How to run the project:
- Clone the repository
- Create a virtual environment
- Install the dependencies
- Run the `application.py` file
- Open the localhost link in the browser
- Enter the values in the form and click on the predict button to get the predicted math score.