'''
Modified 9/2022
Author: ms-analytics
'''
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    """
    This function  takes an argument of a dataset, in a pandas DataFrame format.
    It reads the deployed model from the directory specified in the prod_deployment_path
    entry of config.json file.

    The function uses the deployed model to make predictions for each row of the input 
    dataset. Its output is a list of predictions. 
    **This list should have the same length as the number of rows in the input dataset.
    """
    # Read the deployed model and a test dataset
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), 'rb') as file:
        model = pickle.load(file)

    # read test data
    path = os.path.join(test_data_path, "testdata.csv")
    data = pd.read_csv(path)
    X = data.drop(columns=['corporation', 'exited']).values
    #y = data['exited'].values

    # Calculate predictions
    prediction = model.predict(X)
    return prediction #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    """
    calculates summary statistics on data.
        *means, medians, and standard deviations
    for each numeric column in data and
    stored in the directory specified by 
    output_folder_path in config.json. 

    **It should output a Python list containing 
    all of the summary statistics for every numeric 
    column of the input dataset.
    """
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    numeric_col = data.select_dtypes(include=np.number).columns.tolist()
    X_summary = data[numeric_col].drop(columns = "exited")

    mean = list(X_summary.mean(axis=0).values)
    median = list(X_summary.median(axis=0).values)
    std = list(X_summary.std(axis=0).values)

    summary_statistics = pd.DataFrame(list(zip(X_summary.columns, mean, median, std)),
                              columns=['column', 'mean', 'median', 'std'])

    summary_statistics.to_csv(os.path.join(dataset_csv_path, "summary_df.csv"))
    
    return summary_statistics #return value should be a list containing all summary statistics

def missing_data():
    """
        function to check for missing data (NA values). 
            **Pandas module has a custom method for checking 
            whether a value is NA.

        *counts the number of NA values in each column of dataset. 
        *calculates percent of of NA values.
        *for output_folder_path 
    """
    # read the ingested final data
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    # count the number of NA values in each column for the dataset.
    NA = list(data.isna().sum())
    # calculate what percent of each column consists of NA values.
    percentage = [NA[i] / len(data.index) for i in range(len(NA))]
    result = []
    for i in range(len(data.columns)):
        result.append([data.columns[i], str(percentage[i])+"%"])

    return str(result)
##################Function to get timings
def execution_time():
    """
        function that times how long it takes to perform data ingestion 
        (ingestion.py) and model training (training.py).

        Takes no arguments. 
        returns a Python list consisting of two timing measurements 
        in seconds: one measurement for data ingestion, and one 
        measurement for model training.
    """
    # calculate timing of training.py and ingestion.py
    duration = []
    start = timeit.default_timer()
    os.system('python ingestion.py')
    time1=timeit.default_timer() - start

    duration.append(["ingestion timing", time1])

    start = timeit.default_timer()
    os.system('python training.py')
    time2=timeit.default_timer() - start
    
    duration.append(["training timing", time2])

    return str(duration) #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    '''
        function that checks the current and latest versions of 
        all the modules (the current version is recorded 
        in requirements.txt). It will output a table with three 
        columns: the first column will show the name of a Python 
        module  used; the second column will show the 
        currently installed version, and 
        the third column will show the most recent available 
        version.

        To get the best, most authoritative information about 
        Python modules, you should rely on Python's official 
        package manager, pip. Your script should run a pip 
        command in your workspace Terminal to get the information 
        you need for this step.
    '''
    environment = subprocess.check_output(['pip', 'list', '--outdated'], text = True)
    return str(environment)


if __name__ == '__main__':
    print(model_predictions())
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())





    
