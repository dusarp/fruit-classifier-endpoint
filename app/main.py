from pydantic import BaseModel # this has nothing to do my machine learning models
# this is just the parent class for everything that is strictly typed in Pydantic
from fastapi import FastAPI, Depends, UploadFile, File
from torchvision import transforms
from torchvision.models import ResNet
# we need this to upload images to fastAPI
# this is the Python image library
from PIL import Image
import torch
from app.model import load_model, load_transforms, CATEGORIES
import torch.nn.functional as F
import io
# This is what we use the BaseModel for
# the result is strictly typed so that it returns
# a string for the category (label that we predict)
# and a float for the confidence (the probability for the label)
class Result(BaseModel):
    category: str
    confidence: float
# this creates an instance for the endpoint
app = FastAPI()
# response_model is a pydantic BaseModel, not a machine learning model
# is the POST response that we are defining with class Result(BaseModel)
@app.post('/predict', response_model=Result )
async def predict(
        input_image: UploadFile = File(...),
        model: ResNet = Depends(load_model),
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    # Read the uploaded image
    image = Image.open(io.BytesIO(await input_image.read()))

    # Convert RGBA image to RGB image
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply the transformations
    image = transforms(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # Map the predicted class index to the category
    category = CATEGORIES[predicted_class.item()]

    return Result(category=category, confidence=confidence.item())