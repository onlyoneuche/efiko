import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting the data into train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Select all rows and all columns except the last column
                train_array[:, -1],   # Select all rows and the last column
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            # Hyperparameters for the models
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            model_report = evaluate_models(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=params
            )

            logging.info(f"Model report: {model_report}")
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best model score: {best_model_score}")
            best_model_name = [
                k for k, v in model_report.items() if v == best_model_score
            ][0]
            best_model = models[best_model_name]
            logging.info(f"Model with highest score: {best_model}")
            if best_model_score < 0.6:
                raise CustomException(
                    error_message="Model score is less than 0.6",
                    error_detail=sys,
                )
            save_obj(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model,
            )
            logging.info(f"Saved the best model")

            prediction = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, prediction)
            logging.info(f"R2 score of the best model: {r2_score_value}")
            return r2_score_value

        except Exception as ce:
            raise CustomException(error_message=ce, error_detail=sys)