from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         age = request
#         age = int(age)
#         return render_template('predict.html', age=age)
#
#     return render_template('predict.html')