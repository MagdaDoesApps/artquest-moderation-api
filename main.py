from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

app = FastAPI()

# --- Load Models ---
# 1️⃣ General NSFW/gore model (lightweight YOLOv8n)
nsfw_model = YOLO("yolov8n.pt")

# 2️⃣ Custom trained weapon detection model (ONNX)
weapon_model = YOLO("best.onnx")

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

        # Try opening image
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # --- Run YOLOv8n (general object detection for NSFW cues) ---
        nsfw_results = nsfw_model.predict(img, verbose=False)
        nsfw_labels = [nsfw_model.names[int(b.cls)] for b in nsfw_results[0].boxes]

        # --- Run custom ONNX model for weapon detection ---
        weapon_results = weapon_model.predict(img, verbose=False)
        weapon_labels = [weapon_model.names[int(b.cls)] for b in weapon_results[0].boxes]

        # --- Define unsafe keywords ---
        nsfw_keywords = ["nude", "underwear", "naked", "sexual", "person"]
        weapon_keywords = ["knife", "gun", "rifle", "pistol", "weapon"]

        # --- Determine unsafe content ---
        nsfw_detected = any(any(k in l.lower() for k in nsfw_keywords) for l in nsfw_labels)
        weapon_detected = any(any(k in l.lower() for k in weapon_keywords) for l in weapon_labels)

        unsafe_labels = []
        if nsfw_detected:
            unsafe_labels.append("nsfw/sexual")
        if weapon_detected:
            unsafe_labels.append("weapon")

        return {
            "safe": not (nsfw_detected or weapon_detected),
            "unsafe_reasons": unsafe_labels,
            "nsfw_labels": nsfw_labels,
            "weapon_labels": weapon_labels,
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
