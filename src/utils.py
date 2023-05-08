import os
import sys

import numpy as np
import pandas as pd
import dill
# import pickle5 as pickle

from src.exception import CustomException
from src.logger import logging

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