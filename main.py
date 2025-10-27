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
weapon_model = YOLO("normal.onnx")

class ImageInput(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"message": "✅ ArtQuest Moderation API is online!"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        # Download the image
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # 🔍 Run prediction using your ONNX or YOLOv8 model
        results = model.predict(img, conf=0.25, verbose=False)

        # Extract detected labels
        labels = [model.names[int(b.cls)] for b in results[0].boxes]

        # ✅ Debug print to Render logs
        print("🧠 Detected labels:", labels)

        # Define possible unsafe keywords
        unsafe_keywords = ["gun", "pistol", "rifle", "knife", "weapon", "nude", "blood", "violence"]

        unsafe_detected = any(
            any(k in label.lower() for k in unsafe_keywords)
            for label in labels
        )

        return {
            "safe": not unsafe_detected,
            "labels": labels,
            "unsafe_labels": [l for l in labels if any(k in l.lower() for k in unsafe_keywords)]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

