from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nsfw_detector import predict
from PIL import Image
import requests
from io import BytesIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only warnings and errors
import tensorflow as tf

app = FastAPI()

# Load NSFW model once on startup
model = predict.load_model("model.h5")

class ImageInput(BaseModel):
    image_url: str

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        response = requests.get(data.image_url)
        img = Image.open(BytesIO(response.content))
        result = predict.classify(model, data.image_url)
        print(result)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

