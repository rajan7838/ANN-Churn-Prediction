import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class ModelTrainer:

    def __init__(self):
        self.model_path = 'artifacts/model.h5'
        os.makedirs('artifacts',exist_ok=True)

    def train_model(self,X_train,y_train):

        print("Training ANN model.....")


        model = Sequential([
            Dense(32,activation='relu',input_shape = (X_train.shape[1],)),
            Dense(16,activation='relu'),
            Dense(1,activation='sigmoid')])
        
        model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

        model.fit(X_train,y_train,epochs=10,batch_size=30,verbose=1)

        model.save(self.model_path)

        print("Model Training Complited.....")

        return model
         