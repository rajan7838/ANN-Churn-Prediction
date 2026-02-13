import pandas as pd
import os
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self):
        self.data_path = 'data/Churn_Modelling.csv'
        self.artifacts_dir = 'artifacts'

        os.makedirs('artifacts',exist_ok=True)

    def initiate_data_ingestion(self):
        print("Data Ingestion started....")

        df = pd.read_csv(self.data_path)

        raw_path = os.path.join(self.artifacts_dir,'raw_data.csv')
        df.to_csv(raw_path,index=False)

        train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

        train_path = os.path.join(self.artifacts_dir,'train.csv')
        test_path = os.path.join(self.artifacts_dir,'test.csv')

        train_set.to_csv(train_path,index=False)
        test_set.to_csv(test_path,index=False)

        print("data ingestion is complited....")

        return train_path,test_path






            