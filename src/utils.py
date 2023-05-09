import os
import sys

import numpy as np
import pandas as pd
import dill
# import pickle5 as pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

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
            dill.dump(obj,file_obj)



    except Exception as e:  
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    """
    Function to evaluate the trained model
    """
    try:
        report = {}
        for i in range(len(list(models))):
            # logging.info(f"{i}"

            model = list(models.values())[i]



            model.fit(X_train,y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_model_score = r2_score(y_true=y_train,y_pred=y_pred_train)
            test_model_score = r2_score(y_true=y_test,y_pred=y_pred_test)

            report[list(models.keys())[i]] = test_model_score

        return report
    
     
    except Exception as e:
        raise CustomException(e,sys)