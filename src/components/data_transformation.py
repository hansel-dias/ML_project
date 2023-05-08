import sys
from dataclasses import dataclass

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# Data transformation libraries
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier


# Logging and Exception imports
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Class for saving pipeline.pkl file
    """
        
  
    
    # pipeline_obj_path = os.path.join("D:/ML/trained_models","preprocessor.pkl")
    pipeline_obj_path = os.path.join('trained_models','preprocessor.pkl')
    logging.info(f"{pipeline_obj_path}")


    if os.access(pipeline_obj_path , os.F_OK | os.R_OK):
        logging.info("File exists and is readable.")
    else:
        logging.info("File does not exist or is not readable.")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_datatransformer_obj(self):
        """
        To create all pickle files: which shall transforma categorical features in to  numerical 
         etc. and other feature engineering 

        continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]
        binaries_f = ["Sex", "FastingBS", "ExerciseAngina"]
        """
        try:
            numerical_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]
            categorical_f = ["Sex","ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
            

            # Transformation to be applied to numerical_cols
            numerical_transformer = Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            # Transformation to be applied to catagorical_cols
            categorical_transformer = Pipeline(
                steps=[
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                    ]
            )

            logging.info("Catergorical columns standard scaling completed")

            # prepare transformation
            preprocessor = ColumnTransformer(transformers = [
                    ('num',numerical_transformer,numerical_f),
                    ('cat',categorical_transformer,categorical_f)
            ]
            )
        
            # # Add transformation to pipeline
            # pipeline = Pipeline(steps=[
            #     ('preprocessor',preprocessor),
            #     ('classifier',KNeighborsClassifier())
            # ])

            return preprocessor
        
            
        except Exception as e:
            raise CustomException(e,sys)
            
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        This function reads test,train data, drop target_col from train_df and create target_df
        and then performs data transformation on the data.
        
        """
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test_ data")

                
            preprocessor_obj = self.get_datatransformer_obj()
            target_column_name = "HeartDisease"
            
            # Define train and test dataframe
            input_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_df = train_df[target_column_name]

            input_test_df = test_df.drop(columns = [target_column_name],axis=1)
            target_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
        
            
            """
            Once the scaling parameters have been calculated 
            during the fit_transform stage,the same parameters 
            can be applied to new data using the transform method.
            """
            input_train_arr = preprocessor_obj.fit_transform(input_train_df)  
            input_test_arr = preprocessor_obj.transform(input_test_df)
        

            """
            np.c_ is a function provided by the NumPy library that is used
            to concatenate two arrays column-wise. It creates a new array 
            by stacking 1-D arrays as columns.

            For example, if you have two arrays a and b with shapes (n,) 
            and (m,) respectively, np.c_[a, b] will create a new array of 
            shape (n, 2) by stacking the two arrays column-wise. 
            """    
            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[
                input_test_arr,np.array(target_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.pipeline_obj_path,
                obj = preprocessor_obj
            )
            logging.info(f"Saved preproceessing obj")



            return (
                train_arr,
                test_arr,
                self.data_transformation_config.pipeline_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            


            
            
        
        
