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
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }  # Hyperparameter tuning can be done here as well
            model_report = evaluate_models(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
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