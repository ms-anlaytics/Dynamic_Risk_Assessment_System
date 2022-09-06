
import json
import os
from  diagnostics import model_predictions, execution_time
from  diagnostics import dataframe_summary, missing_data, outdated_packages_list
from itertools import chain
from scoring import score_model
from ingestion import merge_multiple_dataframe
from training import train_model, get_data_path
from deployment import store_model_into_pickle



with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
model_path = os.path.join(config['output_model_path'])

def processing():
##################Check and read new data
#first, read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        current_data = list(chain(*[line.split() for line in file.readlines()]))
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    new_data_list = os.listdir(os.getcwd() + "/" + input_folder_path)


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
    if current_data not in new_data_list:
        print('new data')
        merge_multiple_dataframe()
        score_model()
        
        ##################Checking for model drift
        #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
        with open(os.path.join(model_path, "latestscore.txt"), 'r') as new_score:
            H1 = float(new_score.read())

        with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as old_score:
            H0 = float(old_score.read())

        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        if H1 >= H0:
            print('model drift not found')
            exit(0)

        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        train_model(get_data_path())
        store_model_into_pickle()
        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model

        model_predictions()
        execution_time()
        dataframe_summary()
        missing_data()
        outdated_packages_list()
        score_model()
        print('trained')
    else:
        print('end')
        exit(0)


if __name__ == "__main__":
    processing()






