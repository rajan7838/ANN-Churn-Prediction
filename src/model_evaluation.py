import os
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score
import numpy as np
import json

class ModelEvaluation:

    def evaluate(self,model,X_test,y_test):

        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test,y_pred)
        roc = roc_auc_score(y_test,y_pred)

        print("\n Model Evaluation Results:")
        print("Accuracy:", acc)
        print("ROC-AUC Score:", roc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        os.makedirs("artifacts",exist_ok=True)

        metrics = {
            "accuracy":float(acc),
            "roc_auc":float(roc)
        }

        with open("artifacts/metrics.json",'w') as f:
            json.dump(metrics,f,indent=4)

        print("metrics saved succefully")

        return acc,roc    

    
