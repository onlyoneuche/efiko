import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model = load_obj(file_path="artifacts/model.pkl")
            preprocessor = load_obj(file_path="artifacts/preprocessor.pkl")
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as ce:
            raise CustomException(error_message=ce, error_detail=sys)


class CustomData:
    def __init__(
            self,
            gender,
            race_ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            reading_score,
            writing_score
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self):
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            df = pd.DataFrame(data_dict)
            return df
        except Exception as ce:
            raise CustomException(error_message=ce, error_detail=sys)