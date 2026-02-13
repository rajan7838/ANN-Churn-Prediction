import shutil
import os

class ModelPusher:

    def __init__(self):

        self.model_source = 'artifacts/model.h5'
        self.model_move = 'models/best_model.h5'


        os.makedirs('models',exist_ok=True)

    def push_model(self):
        shutil.copy(self.model_source,self.model_move)
        print("Model pushed successfully to models/ folder!")  


    


        