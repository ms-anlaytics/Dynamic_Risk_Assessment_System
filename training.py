'''
Modified 9/2022
Author: ms-analytics
'''

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
def get_data_path():
    '''
        Gets data path from config
    '''
    with open('config.json','r') as f:
        config = json.load(f) 
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    
    return os.path.join(dataset_csv_path, "finaldata.csv")

def set_model_path():
    '''
        Sets model path from config
    '''
    with open('config.json','r') as f:
        config = json.load(f) 
    model_path = os.path.join(config['output_model_path'])
    return model_path

def write_model(model):
    '''
        Writes model to model path
    '''
    filehandler = open(os.path.join(set_model_path(), "trainedmodel.pkl"), 'wb')
    pickle.dump(model, filehandler)

#################Function for training the model
def train_model(path):
    # read data
    data = pd.read_csv(path)
    X = data.drop(columns=['corporation', 'exited']).values
    y = data['exited'].values
    
    #use this logistic regression for training
    logit =LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
     # fit the logistic regression to your data
    model = logit.fit(X, y)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    write_model(model)
    



if __name__ == "__main__":
    train_model(get_data_path())


