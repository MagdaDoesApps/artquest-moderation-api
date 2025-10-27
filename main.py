from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests
import os

app = FastAPI()

# -------------------------------
# Load models
# -------------------------------
nsfw_model = YOLO("yolov8n.pt")   # Tiny COCO model (for person/nude)
weapon_model = YOLO("best.pt")    # Weapons/knife detector

class ImageInput(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"message": "✅ ArtQuest Moderation API is online!"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        # Fetch image
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # -------------------------------
    # Run NSFW detection (using COCO model)
    # -------------------------------
    nsfw_results = nsfw_model.predict(img, verbose=False)
    nsfw_labels = [nsfw_model.names[int(b.cls)] for b in nsfw_results[0].boxes]
    nsfw_keywords = ["nude", "person"]
    nsfw_detected = any(k in l.lower() for l in nsfw_labels for k in nsfw_keywords)

    # -------------------------------
    # Run weapons detection
    # -------------------------------
    weapon_results = weapon_model.predict(img, verbose=False)
    weapon_labels = [weapon_model.names[int(b.cls)] for b in weapon_results[0].boxes]
    weapon_keywords = ["knife", "gun", "weapon", "blade"]
    weapon_detected = any(k in l.lower() for l in weapon_labels for k in weapon_keywords)

    # -------------------------------
    # Combine results
    # -------------------------------
    unsafe_reasons = []
    if nsfw_detected:
        unsafe_reasons.append("nsfw/sexual")
    if weapon_detected:
        unsafe_reasons.append("weapon")

    return {
        "safe": not unsafe_reasons,
        "unsafe_reasons": unsafe_reasons,
        "nsfw_labels": nsfw_labels,
        "weapon_labels": weapon_labels
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
