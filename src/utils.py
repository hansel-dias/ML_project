import os
import sys

import numpy as np
import pandas as pd
# import dill
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score,recall_score,classification_report
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
def save_object(file_path,obj):
    """
    Function to save pickle file. this function two 
    arguments file path and pickle obj 
    """

    try:
        # logging.info(f"{file_path}")

        obj_path  = os.path.dirname(file_path)
        os.makedirs(obj_path,exist_ok=True)
        # assert os.path.isfile(obj_path)
        # logging.info(f"{os.path.isfile(obj_path)}")
        # logging.info(f"{obj_path}")
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)



    except Exception as e:  
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    """
    Function to evaluate the trained model
    """
    try:
        report = {}
        for i in range(len(list(models))):
            # logging.info(f"{i}"

            # model = list(models.values())[i]
            # param = params[list(models.keys())[i]]
            
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[model_name]


            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)


            # model.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_model_score = recall_score(y_true=y_train,y_pred=y_pred_train)
            test_model_score = recall_score(y_true=y_test,y_pred=y_pred_test)

            report[model_name] = test_model_score

        return report
    
     
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)