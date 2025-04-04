import dagshub
dagshub.init(repo_owner='Aman-3110', repo_name='AirlineFare_EndToEnd', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)