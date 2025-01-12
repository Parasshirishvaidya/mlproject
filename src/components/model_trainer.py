#   Importing Necessary libraries 
import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

# Creating Model Config Class
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

# Creating Model Trainer Class
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        # Try and Cathc Block
        try:
            logging.info("Splitting training and testing data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier" : CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostRegressor(),
            }

            param_grids = {
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
                "Decision Tree": {"max_depth": [10, 20, None], "criterion": ["squared_error", "friedman_mse"]},
                "Gradient Boosting": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [100, 200]},
                "Linear Regression": {},  # No hyperparameters to tune for LinearRegression
                "K-Neighbors Classifier": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                "XGBClassifier": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
                "CatBoosting Classifier": {"depth": [6, 8], "learning_rate": [0.01, 0.1], "iterations": [100, 200]},
                "AdaBoost Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.5]},
                }
            
            model_report, best_model_param = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models, 
                params=param_grids
                )

            # to get the best model score from the Dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from the Dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best Model Found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test,predicted),best_model_param,best_model_name
            return score,best_model_name,best_model.get_params()
        except Exception as e:
            raise CustomException(e,sys)
