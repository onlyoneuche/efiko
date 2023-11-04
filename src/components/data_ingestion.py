import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """
    This is the dataclass for DataIngestionConfig
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    DATA_PATH = "src/notebook/data/stud.csv"

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function is used to initiate the data ingestion process
        :return: None
        """
        logging.info("Starting the data ingestion process")
        try:
            df = pd.read_csv(self.DATA_PATH)
            logging.info(f"Data ingested as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info(f"Starting train test split")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train test split completed and saved")
        except CustomException as ce:
            logging.info(f"Error in reading the data")
            raise CustomException(error_message=ce, error_detail=sys)

        return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path=train_data, test_data_path=test_data)
