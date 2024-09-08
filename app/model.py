import wandb
import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet
#ez kezeli a wandb api secret tokent 


from loadotenv import load_env
load_env(file_loc='/workspaces/fruit-classifier-endpoint/app/.env')



MODELS = 'models'
MODEL_FILE_NAME = 'model.pth'
CATEGORIES = ["freshapples", "freshbanana", "freshoranges",
              "rottenapples", "rottenbanana", "rottenoranges"]

#delete this
print(os.getenv('WANDB_API_KEY')) 

def download_artifact():
    assert "WANDB_API_KEY" in os.environ, "Please enter the required environment variables."

    wandb.login()
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")

    artifact_path = "dusarp-dsr/'banana_apple_orange'/resnet18:v1"
    artifact = wandb.Api().artifact(artifact_path, type="model")
    artifact.download(root='MODELS_DIR')


download_artifact()