'''
Modified 9/2022
Author: ms-analytics
'''

import os
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    '''copies trained model (trainedmodel.pkl), 
    your model score (latestscore.txt), and a record of 
    your ingested data (ingestedfiles.txt). It copies 
    all three of these files from their original locations 
    to a production deployment directory. The location of 
    the production deployment directory is specified in the 
    prod_deployment_path entry of config.json file.''' 

    model = os.path.join(output_model_path, "trainedmodel.pkl")
    model_score = os.path.join(output_model_path, "latestscore.txt")
    ingested_data = os.path.join("ingestedfiles.txt")

    source = [model, model_score, ingested_data]
    deployment_directory = prod_deployment_path

    # Copy the content of source to destination
    for s in source:
        dest = shutil.copy(s, deployment_directory)


if __name__ == "__main__":
    store_model_into_pickle()       

