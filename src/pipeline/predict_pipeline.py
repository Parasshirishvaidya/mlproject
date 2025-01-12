import sys 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            # Load the model
            model = load_object(file_path=model_path)
            # Define columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]
            # Define pipelines
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),]
                    )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),])
            
            # Create ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ])
            
            # Load training data and fit preprocessor
            train_data = pd.read_csv("artifacts/train.csv")
            preprocessor.fit(train_data)

            print(features)
            # Transform input features
            data_scaled = preprocessor.transform(features)


            # Make predictions
            preds = model.predict(data_scaled)

            # Return predictions
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
            gender:str,
            race_ethnicity:str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            reading_score:int,
            writing_score:int):
        
        self.gender =gender
        
        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
    
        except Exception as e :
            raise CustomException(e,sys)
            
