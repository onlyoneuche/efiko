import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application= Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        df = data.get_data_as_df()
        print("df: ", df)
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(df)
        print("result: ", result)
        return render_template('home.html', result=result[0])


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)
