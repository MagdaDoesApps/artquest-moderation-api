from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

# NSFW model
from nsfw_detector import predict

app = FastAPI()

# ---- Load models once ----
weapon_model = YOLO("yolov8n.pt")          # General object detector (weapons, blood, gore)
nsfw_model = predict.load_model("nsfw_model.h5")  # Real NSFW model

class ImageInput(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"message": "✅ ArtQuest Moderation API is online!"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        # Fetch image bytes
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # ---- YOLO weapon/gore detection ----
    results = weapon_model.predict(img, verbose=False)
    labels = [weapon_model.names[int(b.cls)] for b in results[0].boxes]

    weapon_keywords = ["knife", "gun", "weapon", "blood"]
    weapon_detected = any(k in l.lower() for k in labels for k in weapon_keywords)

    # ---- NSFW detection ----
    nsfw_scores = predict.classify(nsfw_model, img)
    # nsfw_scores is a dict like {"image_path": {"neutral":0.1,"porn":0.8,"sexy":0.1,"hentai":0.0,"drawings":0.0}}
    nsfw_result = list(nsfw_scores.values())[0]
    nsfw_detected = nsfw_result.get("porn",0) + nsfw_result.get("sexy",0) + nsfw_result.get("hentai",0) > 0.3

    # ---- Combine results ----
    unsafe = weapon_detected or nsfw_detected
    unsafe_reasons = []
    if weapon_detected:
        unsafe_reasons.append("weapon/gore")
    if nsfw_detected:
        unsafe_reasons.append("nsfw/sexual")

    return {
        "sa
