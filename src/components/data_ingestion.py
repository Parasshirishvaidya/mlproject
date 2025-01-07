import os
import sys 
from src.exception import CustomException
from src.logger import logging 

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data Ingestion Component')
        try:
            # Reading data from the source as a DataFrame
            df=pd.read_csv('Notebook\data\stud.csv')
            logging.info("Read the Dataset as a DataFrame")

            # Creating Artifacts folder pass if already exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Creating or updating raw data file 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
             
            # Creating training and testing set
            logging.info("Train Test Split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # Creating train and test csv files in the Artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            train_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()