'''
Updated 9/2022
Author: ms-anlaytics
'''

import pandas as pd
import os
import json





#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

'''When we're initially setting up the project, 
our config.json file will be set to read practicedata 
and write practicemodels. When we're ready to finish 
the project, you will need to change the locations 
specified in config.json so that we're reading our 
actual, sourcedata and we're writing to our models directory.'''


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file

    filenames = os.listdir(os.getcwd() + "/" + input_folder_path)
    data_list = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity',
                                    'number_of_employees', 'exited'])
    for each_filename in filenames:
        
        new_file = pd.read_csv(os.path.join(os.getcwd(), input_folder_path, each_filename))
        data_list = pd.concat((data_list, new_file), axis=0)
        with open('ingestedfiles.txt', 'a') as f:
            f.write(f"{each_filename} \n")

    result = data_list.drop_duplicates()
    output_pth = os.path.join(output_folder_path, "finaldata.csv")
    result.to_csv(output_pth, index=False)
    return result

if __name__ == '__main__':
    merge_multiple_dataframe()
