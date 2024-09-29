from flask import Flask, request, render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, Predict_Pipeline

application = Flask(__name__)

app = application

#Route for Home
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods= ['GET', 'post'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )
        
        pred_df = data.get_data_as_DataFrame()
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_rows", None)
        print(pred_df)
        print("Columns:::::::::::::: \n", pred_df.columns)
        
        predict_pipeline = Predict_Pipeline()
        results = predict_pipeline.predict(pred_df)
        # results = predict_pipeline.predict(pred_df.dropna(axis=0))
        
        return render_template('home.html',results=results,pred_df = pred_df)
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug= True)