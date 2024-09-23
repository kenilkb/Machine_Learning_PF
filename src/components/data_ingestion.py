import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
from src import utils

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("from Data ingestion Component..")
        try:
            #Locate Data Here, to get ready-made dataFrame
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            logging.info("Train_test_split initiated...")
            train_set, test_set = train_test_split(df, random_state = 42, test_size = 0.2)
            
            train_set.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("data Ingestion completed!")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise customException(e,sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformer = DataTransformation()
    train_array, test_array, _ = data_transformer.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()

    print("Trained with... ",modeltrainer.initiate_model_trainer(train_array, test_array) , " R2")
    
    
    
        
        

