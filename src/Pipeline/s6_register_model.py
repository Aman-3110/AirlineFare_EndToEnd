import os
import json
import mlflow


class ModelManager:
    """Class to manage model saving, loading, and registration with MLflow."""

    def __init__(self, model_name: str, info_path: str):
        self.model_name = model_name
        self.info_path = info_path
        self.client = mlflow.tracking.MlflowClient()

    def save_model_info(self, run_id: str, model_path: str) -> None:
        """Save the model run ID and path to a JSON file."""
        os.makedirs(os.path.dirname(self.info_path), exist_ok=True)  # Ensure the directory exists
        model_info = {'run_id': run_id, 'model_path': model_path}
        
        with open(self.info_path, 'w') as file:
            json.dump(model_info, file, indent=4)

    def load_model_info(self) -> dict:
        """Load the model info from a JSON file."""
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"Model info file not found: {self.info_path}")

        with open(self.info_path, 'r') as file:
            return json.load(file)

    def register_model(self):
        """Register the model with the MLflow Model Registry."""
        model_info = self.load_model_info()
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(model_uri, self.model_name)


        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version.version,
            stage="Staging"
        )
        

if __name__ == '__main__':
    model_name="my_model"
    info_path='reports/experiment_info.json'

    model_manager = ModelManager(model_name, info_path)
    
    # Load model info and register
    model_manager.register_model()