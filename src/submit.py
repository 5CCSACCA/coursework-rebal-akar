from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration
import os

# Optionally load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Connect to workspace using config.json
ws = Workspace.from_config()

# Define the experiment
experiment = Experiment(workspace=ws, name='distilbert-hatespeech-distributed')

# Define compute target
compute_name = "distilberthatespeech"

try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print(f'Found existing compute target: {compute_name}')
except ComputeTargetException:
    print(f'Creating new compute target: {compute_name}')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_E4ds_v4', max_nodes=5)
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Define environment
env = Environment.from_conda_specification(name='mlenv', file_path='env.yml')

# Retrieve MLflow tracking URI
mlflow_tracking_uri = ws.get_mlflow_tracking_uri()

# Define PyTorch distributed configuration with corrected parameter names
pytorch_config = PyTorchConfiguration(
    node_count=5,                  # Number of nodes
)



# Define ScriptRunConfig with distributed configuration
src = ScriptRunConfig(
    source_directory='.',  # Directory containing model.py and other files
    script='model.py',
    compute_target=compute_target,
    environment=env,
    arguments=[
        '--epochs', '5',
        '--learning_rate', '2e-5',
        '--batch_size', '8',
        '--accumulation_steps', '2'
    ],
    distributed_job_config=pytorch_config  # Assign the PyTorch configuration
)



src.run_config.environment_variables = {
    'STORAGE_ACCOUNT_KEY': 'Z847O5qPcdxoV4JznEmYRKcZ72LAAP9Yz6zGAPDS6qjD8aDBM1IqaznDQMyXSsTVvZP/CGO/upAF+AStWqlXJg==',  # Replace with your actual key
    'STORAGE_ACCOUNT_CONNECTION_STRING': 'DefaultEndpointsProtocol=https;AccountName=cloudcomputing8991014366;AccountKey=Z847O5qPcdxoV4JznEmYRKcZ72LAAP9Yz6zGAPDS6qjD8aDBM1IqaznDQMyXSsTVvZP/CGO/upAF+AStWqlXJg==;EndpointSuffix=core.windows.netyour_storage_account_connection_string',  # Replace with your actual connection string
    'MLFLOW_TRACKING_URI': mlflow_tracking_uri,
    'MLFLOW_S3_ENDPOINT_URL': 'https://cloudcomputing8991014366.blob.core.windows.net/'  # Replace with your Blob Storage endpoint

}

# Submit the experiment
run = experiment.submit(src)
run.wait_for_completion(show_output=True)

# Register the model based on performance
metrics = run.get_metrics()
if metrics.get('test_f1') and metrics['test_f1'] > 0.8:  # Example condition
    model = run.register_model(model_name='distilbert_hatespeech_model', 
                               model_path='distilbert_hatespeech_model')
    print(f"Model registered: {model.name} with version {model.version}")
