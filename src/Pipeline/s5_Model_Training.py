import pandas as pd
import os
import yaml
import numpy as np 
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
#from src.Utils.Utils import load_yaml, get_class, get_class_Scaler
import json

import mlflow
import dagshub

class ModelTrainerClass:
    def __init__(self, X_train_path, y_train_path, params_path):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.params_path = params_path
        
        # self.load_yaml = load_yaml
        # self.data = self.load_yaml(yaml_path="constants.yaml")
        # self.CrossValidation =  self.data['trainTestSplit']['CrossValidation']
        
    def load_X_train(self):
        X_train = pd.read_csv(self.X_train_path)
        self.X_train = np.array(X_train)
        return self.X_train
    
    def load_y_train(self):
        y_train = pd.read_csv(self.y_train_path)
        self.y_train = np.array(y_train).ravel()
        return self.y_train
    
    def load_params(self):
        with open(self.params_path, 'r') as file:
            self.modelwithParams = yaml.safe_load(file)
            self.modelwithParams = self.modelwithParams['model']

        return self.modelwithParams
            
    def get_Model_class(self, model):
        model_class = globals()[model.split('.')[-1]]
        model = model_class()
        return model 
    
    def train_model(self, model, param_grid):
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5)
        random_search.fit(self.X_train, self.y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        best_model.fit(self.X_train, self.y_train)
        predicted_y = best_model.predict(self.X_train)

        return best_model, best_params, predicted_y
    
    def calculate_metrics(self, actual_X, actual_y, predicted_y):
        mse = mean_squared_error(actual_y, predicted_y)
        mae = mean_absolute_error(actual_y, predicted_y)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_y, predicted_y)

        n = len(actual_y)
        p = actual_X.shape[1]    
        ar2 = 1 - (1-r2)*(n-1)/(n-p-1)

        return mse, mae, rmse, r2, ar2
    
    def get_Model_Name(self, model):
        model_name = str(model.__class__.__name__)
        return model_name
    

class MLflowLoggerClass:
    def __init__(self):
        mlflow.set_tracking_uri('https://dagshub.com/Aman-3110/AirlineFare_EndToEnd.mlflow')
        dagshub.init(repo_owner='Aman-3110',repo_name='AirlineFare_EndToEnd',mlflow=True)

        # self.tracking_uri = tracking_uri
        # mlflow.set_tracking_uri(self.tracking_uri)


    def save_model_info(self, run_id: str, model_path: str, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_info = {'run_id': run_id, 'model_path': model_path}

        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
                        


    def  log_results(self, model_name, best_model, best_params, mse, mae, rmse, r2, ar2):
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("Adjusted R2", ar2)
                              
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"Best {param_name}", param_value)

            mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            print(run.info.run_id)

            self.save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')


            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print(f"R2: {r2}")
            print(f"Adjusted R2: {ar2}")
            



if __name__ == "__main__":
    X_train_path = 'Data/04_Encoded_Data/X_train.csv'
    y_train_path = 'Data/04_Encoded_Data/y_train.csv'

    params_path = "modelsParams.yaml"
    tracking_uri = "https://dagshub.com/Aman-3110/AirlineFare_EndToEnd.mlflow"

    ModelTrainerObj = ModelTrainerClass(X_train_path, y_train_path, params_path)

    X_train = ModelTrainerObj.load_X_train()
    y_train = ModelTrainerObj.load_y_train()
    modelWithParams = ModelTrainerObj.load_params()
    print("------------------------------")
    
    
    for value in modelWithParams.values():
        model = value['model']
        model = ModelTrainerObj.get_Model_class(model)
        params_Grid = value['param']

        print(model)
        print(params_Grid)

        best_model, best_params, predicted_y = ModelTrainerObj.train_model(model, params_Grid)

        mse, mae, rmse, r2, ar2 = ModelTrainerObj.calculate_metrics(X_train, y_train, predicted_y)

        model_name = ModelTrainerObj.get_Model_Name(model)
        print(f"Model Name: {model_name}")

        MLFlowLoggerObj = MLflowLoggerClass()
        MLFlowLoggerObj.log_results(model_name, best_model, best_params, mse, mae, rmse, r2, ar2)




