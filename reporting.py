'''
Modified 9/2022
Author: ms-analytics
'''

import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from diagnostics import model_predictions
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##############Function for reporting
def score_model():
    """
    confusion matrix using the test data and the deployed model
    calls the model_prediction() function in diagnostics.py
    
    """
    # importing test data and the deployed model
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), 'rb') as file:
        model = joblib.load(file)

    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y = test_data['exited'].values

    pred = model_predictions()

    # write the confusion matrix to the workspace
    plt.figure(figsize=[10, 10])
    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(model_path,'confusionmatrix.png'))
   


if __name__ == '__main__':
    score_model()
