import mlflow
import mlflow.pyfunc
import json
from flask import Flask, request, render_template
from flask_cors import cross_origin
import os

from src.Prediction.predictionFile import ReceiveData
import dagshub

app = Flask(__name__)

mlflow.set_tracking_uri('https://dagshub.com/Aman-3110/AirlineFare_EndToEnd.mlflow')
dagshub.init(repo_owner='Aman-3110',repo_name='AirlineFare_EndToEnd',mlflow=True)


def load_model_info() -> dict:
    """Load the model info from a JSON file."""
    if not os.path.exists('reports/experiment_info.json'):
        raise FileNotFoundError(f"Model info file not found: {'reports/experiment_info.json'}")

    with open('reports/experiment_info.json', 'r') as file:
        return json.load(file)

ModelName = "Lasso"
model_info = load_model_info()
model_uri = f"runs:/{model_info['run_id']}/{ModelName}"



try:
    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(model_uri)

    print(model)
except Exception as e:
    print("Error loading model:", e)

@app.route("/")
@cross_origin()

def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()

def predict():
    if request.method=="GET":
        return render_template(home.html)
    
    elif request.method=="POST":

        receiveData_Obj =  ReceiveData()
        
        df = receiveData_Obj.receive_data_from_ui_create_df(Airline = request.form.get('Airline'),
                                        Date_of_Journey = request.form.get('Date_of_Journey'),
                                        Source = request.form.get('Source'),
                                        Destination = request.form.get('Destination'),
                                        Dep_Time = request.form.get('Dep_Time'),
                                        Arrival_Time = request.form.get('Arrival_Time'),
                                        Duration = request.form.get('Duration'),
                                        Total_Stops = request.form.get('Total_Stops'))
        print(df)

        value = receiveData_Obj.execute_pipeline(df)
        prediction_value = model.predict(value)
        print("----------------------------------------------------")
        print(prediction_value)
        print("----------------------------------------------------")
        return render_template("prediction.html", prediction=prediction_value)

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)