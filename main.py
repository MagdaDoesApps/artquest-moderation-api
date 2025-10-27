from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

app = FastAPI()

# Load YOLO models once
nsfw_model = YOLO("yolov8n.pt")       # Tiny NSFW/general model
weapons_model = YOLO("best.pt")       # Weapons & knives model

class ImageInput(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"message": "✅ ArtQuest Moderation API is online!"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        # Fetch the image bytes
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()

        # Open image with Pillow
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # Run NSFW model prediction
        nsfw_results = nsfw_model.predict(img, verbose=False)
        nsfw_labels = [nsfw_model.names[int(box.cls)] for box in nsfw_results[0].boxes]

        # Run Weapons model prediction
        weapons_results = weapons_model.predict(img, verbose=False)
        weapons_labels = [weapons_model.names[int(box.cls)] for box in weapons_results[0].boxes]

        # Combine labels
        all_labels = nsfw_labels + weapons_labels

        # Define unsafe keywords (NSFW + weapons)
        unsafe_keywords = ["knife", "gun", "weapon", "blood", "nude", "person"]

        # Check if any unsafe label is detected
        unsafe_labels = [l for l in all_labels if any(k in l.lower() for k in unsafe_keywords)]
        is_safe = len(unsafe_labels) == 0

        return {
            "safe": is_safe,
            "labels": all_labels,
            "unsafe_labels": unsafe_labels
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)
