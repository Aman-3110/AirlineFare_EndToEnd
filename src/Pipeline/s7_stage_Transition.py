import os
import json

import mlflow
import time
import dagshub




class ModelManager:
    """Class to manage model saving, loading, and registration with MLflow."""

    def __init__(self, model_name: str, info_path: str):
        self.model_name = model_name
        self.info_path = info_path
        self.client = mlflow.tracking.MlflowClient()
        
        mlflow.set_tracking_uri('https://dagshub.com/Aman-3110/AirlineFare_EndToEnd.mlflow')

        dagshub.init(repo_owner='Aman-3110',repo_name='AirlineFare_EndToEnd',mlflow=True)
        # self.tracking_uri = "http://localhost:5000"
        # mlflow.set_tracking_uri(self.tracking_uri)


    def load_model_info(self) -> dict:
        """Load the model info from a JSON file."""
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"Model info file not found: {self.info_path}")

        with open(self.info_path, 'r') as file:
            return json.load(file)

    def register_model(self):
        model_info = self.load_model_info()
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            run_id=model_info['run_id']
        ).version

        # Transition to Staging
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )

if __name__ == '__main__':
    model_name="Lasso_model"
    info_path='reports/experiment_info.json'

    model_manager = ModelManager(model_name, info_path)
    
    # Load model info and register
    model_manager.register_model()