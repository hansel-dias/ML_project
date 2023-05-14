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

from sklearn.metrics import r2_score,recall_score

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

            params={
                
                "Random Forest":{
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting":{
                    "learning_rate": [0.01, 0.1, 1],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20]
                },
                "K-Neighbours":{
                    "n_neighbors": [3, 5, 10],
                    "leaf_size": [10, 30, 50]
                    },
                "XGBClassifier":{
                    "learning_rate": [0.01, 0.1, 1],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
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
                                                 models=models,params=params)
       
            # best_model_score = max(sorted(model_report.values()))
            ####################################
            best_model = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model]

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

            recall = recall_score(y_test,predictions)

            return recall
        except Exception as e:
            raise CustomException(e,sys)