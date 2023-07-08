import os
import sys
from typing import Generator, List, Tuple

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.constant import *

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Random Forest': RandomForestClassifier(n_jobs=-1),
                #'XGBClassifier': XGBClassifier(n_jobs=-1),
                #'SVC': SVC(),
            }
            params = {
                'Random Forest': {
                   # 'n_estimators': [100, 200, 300],
                },
                # 'XGBClassifier': {
                #     'n_estimators': [100, 200, 300],
                # },
                # 'SVC': {
                #     'C': [1, 10, 100],
                #     'kernel': ['linear', 'rbf'],
                #     'gamma': ['scale', 'auto']
                # },
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", error_detail=sys.exc_info()[1] if sys.exc_info()[1] is not None else None)

            logging.info("Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            return accuracy_score(y_test, predicted)
        except Exception as e:
            raise CustomException(e, sys) from e
