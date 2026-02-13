import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessing:

    def __init__(self):
        self.artifact_dir = 'artifacts'
        os.makedirs('artifacts',exist_ok=True)

    def data_preprocessing(self,train_path,test_path):
        print("data_preprocessing started.....")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df = train_df.drop(["RowNumber","CustomerId","Surname"],axis=1)
        test_df = test_df.drop(["RowNumber","CustomerId","Surname"],axis=1)

        X_train = train_df.drop("Exited",axis=1)
        y_train = train_df['Exited']

        X_test = test_df.drop("Exited",axis=1)
        y_test = test_df['Exited']


        # Encode Gender

        label_encoder = LabelEncoder()
        X_train['Gender'] = label_encoder.fit_transform(X_train['Gender'])
        X_test['Gender'] = label_encoder.transform(X_test['Gender'])

        # OneHot Encode Geography
        onehot_encoder = OneHotEncoder(sparse_output=False)

        geo_encoded_train = onehot_encoder.fit_transform(X_train[["Geography"]])
        geo_encoded_test = onehot_encoder.transform(X_test[["Geography"]])

        # Convert to DataFrame
        geo_train = pd.DataFrame(
        geo_encoded_train,
        columns=onehot_encoder.get_feature_names_out(["Geography"]))

        geo_test = pd.DataFrame(
        geo_encoded_test,
        columns=onehot_encoder.get_feature_names_out(["Geography"]))

       # Reset index
        X_train = X_train.drop("Geography", axis=1).reset_index(drop=True)
        X_test = X_test.drop("Geography", axis=1).reset_index(drop=True)

        geo_train = geo_train.reset_index(drop=True)
        geo_test = geo_test.reset_index(drop=True)

        # Concatenate
        X_train = pd.concat([geo_train, X_train], axis=1)
        X_test = pd.concat([geo_test, X_test], axis=1)


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        pickle.dump(scaler, open("artifacts/scaler.pkl", "wb"))
        pickle.dump(label_encoder, open("artifacts/label_encoder.pkl", "wb"))
        pickle.dump(onehot_encoder, open("artifacts/onehot_encoder.pkl", "wb"))

        print("Data Preprocessing Completed!")

        return X_train, X_test, y_train, y_test
                       



