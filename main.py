from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import requests
from transformers import pipeline
from ultralytics import YOLO

app = FastAPI()

# Load both models
nsfw_model = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
yolo_model = YOLO("yolov8n.pt")  # for violence/weapons

class ImageInput(BaseModel):
    image_url: str

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        response = requests.get(data.image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Run NSFW model
        nsfw_result = nsfw_model(img)
        nsfw_score = next((r['score'] for r in nsfw_result if r['label'].lower() == 'nsfw'), 0)

        # Run YOLO for weapons/violence
        detections = yolo_model(img)
        weapon_score = 0
        for det in detections[0].boxes.data.tolist():
            cls_id = int(det[5])
            label = yolo_model.names[cls_id].lower()
            if any(w in label for w in ["knife", "gun", "weapon"]):
                weapon_score = max(weapon_score, float(det[4]))

        safe = nsfw_score < 0.3 and weapon_score < 0.3

        return {
            "safe": safe,
            "nsfw_score": nsfw_score,
            "weapon_score": weapon_score,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
