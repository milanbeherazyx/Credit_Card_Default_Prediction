import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    try:
        model_result = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(model_result)
    except CustomException as e:
        logging.error(f"An error occurred during model training: {e}")
