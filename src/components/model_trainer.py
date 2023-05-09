import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelConfig:
    model_path = os.path.join('trained_models','model.pkl')

class  ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Starting Model Trainer")
            logging.info("Spliting train and test data ")
            X_train, y_train,X_test,y_test = (train_arr[:,:-1], # all column except last
                                                train_arr[:,-1], # only last Column
                                                test_arr[:,:-1],
                                                test_arr[:,-1])
            
            # Dictionary of model to be tried out.

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbours": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost": CatBoostClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }
            model_report : dict = evaluate_models(X_train=X_train,y_train=y_train,
                                                 X_test=X_test,y_test=y_test,
                                                 models=models)
       
            best_model_score = max(sorted(model_report.values()))
            # From 
            best_model_name = list(model_report.keys())[
                            list(model_report.values()).index(best_model_score)]    
            
            best_model = models[best_model_name]

            logging.info("Model Trainer Completed")

            if best_model_score < 0.6:
                # raise CustomException("No best model found",sys)
                logging.info("No best model found")
            else:
                logging.info(f'best model found on both traing and test dataset {best_model}')


            logging.info(f"{model_report}")
            logging.info(f"{best_model}")
            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            
            )
            logging.info('model saved to pkl file')


            predictions = best_model.predict(X_test)

            r2_square = r2_score(y_test,predictions)

            return r2_square
        except Exception as e:
            raise CustomException(e,sys)