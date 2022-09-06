'''
Modified 9/2022
Author: ms-analytics
'''

import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # load trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    X = df.drop(columns = ['corporation', 'exited']).values
    y = df['exited'].values

    predict=model.predict(X)
    f_score=f1_score(predict, y)

    # Write f1_score to latestscore.txt file
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"{f_score}\n")
    return f_score


if __name__ == "__main__":
    score_model()

