import os 
import sys 

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
# Creating Evaluate Model Function
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]] 

            # Finding the best parameters
            gs= GridSearchCV(model,param)
            gs.fit(X_train,y_train)

            #setting the model to those parameter and training it
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        # Retriving best model name and best model parameters 
        best_model_score = max(report.values())  # Get the highest score
        best_model_name = [key for key, value in report.items() if value == best_model_score][0]
        best_model_param = params[best_model_name]

        return report,best_model_param
    except Exception as e:
        raise CustomException(e,sys)
    
# Function to load a file 
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        