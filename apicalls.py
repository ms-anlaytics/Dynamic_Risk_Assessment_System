'''
Modified 9/2022
Author: ms-analytics
'''

import requests
import os
import json


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

with open('config.json','r') as f:
    config = json.load(f)
output_model_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
respond1 = requests.get('http://127.0.0.1:8000/prediction').text
respond2 = requests.get('http://127.0.0.1:8000/scoring').text
respond3 = requests.get('http://127.0.0.1:8000/summarystats').text
respond4 = requests.get('http://127.0.0.1:8000/diagnostics').text

#combine all API responses
responses =  {
                "prediction"  : respond1,
                "scoring"     : respond2,
                "summarystats": respond3,
                "diagnostics": respond4
                }

#write the responses to your workspace
with open(os.path.join(output_model_path, 'apireturns.txt'), 'a') as file:
    
    file.write(str(respond1) + "\n")
    file.write(str(respond2) + "\n")
    file.write(str(respond3) + "\n")
    file.write(str(respond4) + "\n")



    








