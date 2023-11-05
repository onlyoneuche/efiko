import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        This function is used to get the data transformer object
        :return:
        """
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))
                ]
            )
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Created the numerical and categorical transformers")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_features),
                    ("cat", categorical_transformer, categorical_features)
                ]
            )
            return preprocessor
        except Exception as ce:
            raise CustomException(error_message=ce, error_detail=sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info(f"Train and Test Data ingested as dataframes")

            logging.info(f"Fetching preprocessor object")
            preprocessor = self.get_data_transformer_obj()
            target_feature = "math_score"

            input_feature_train_df = train_df.drop(target_feature, axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(target_feature, axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info(f"Applying preprocessor on train and test data")

            input_feature_train_array = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info(f"Saving the preprocessor object")
            save_obj(file_path=self.data_transformation_config.preprocessor_file_path, obj=preprocessor)
            return train_array, test_array, self.data_transformation_config.preprocessor_file_path

        except Exception as ce:
            raise CustomException(error_message=ce, error_detail=sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path=train_data, test_data_path=test_data)

