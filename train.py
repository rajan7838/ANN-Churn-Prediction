from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_evaluation import ModelEvaluation
from src.model_pusher import ModelPusher
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
        
        print("\n ANN Churn Prediction Training Pipeline Started...\n")

        ingestion = DataIngestion()
        train_path,test_path = ingestion.initiate_data_ingestion()

        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.data_preprocessing(train_path,test_path)

        trainer = ModelTrainer()
        model = trainer.train_model(X_train,y_train)


        pusher = ModelPusher()
        pusher.push_model()


        evaluator = ModelEvaluation()
        evaluator.evaluate(model, X_test, y_test)

        print("\n Training Pipeline Completed Successfully!")



