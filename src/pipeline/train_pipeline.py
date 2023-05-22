import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import os






class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("trained_models","model.pkl")
            preprocessor_path=os.path.join('preprocessor','preprocessor_obj.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            logging.info(f"featr:{features}")
            data_scaled=preprocessor.transform(features)
            logging.info(f"scal:{data_scaled}")

            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)