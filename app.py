'''
Modified 9/2022
Author: ms-analytics
'''
from flask import Flask
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
#app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    
    return str(model_predictions()) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats1():        
    #check the score of the deployed model
    return str(score_model()) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats2():        
    #check means, medians, and modes for each column
    return str(dataframe_summary())  #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats3():        
    #check timing and percent NA values
    return str(execution_time()) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
