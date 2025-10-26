from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
from io import BytesIO
import requests

app = FastAPI()

# ✅ lightweight NSFW detection model
nsfw_model = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

class ImageInput(BaseModel):
    image_url: str

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        img = Image.open(BytesIO(requests.get(data.image_url).content))
        results = nsfw_model(img)
        nsfw_score = next((r['score'] for r in results if r['label'].lower() == "nsfw"), 0.0)
        return {"safe": nsfw_score < 0.3, "nsfw_score": nsfw_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
