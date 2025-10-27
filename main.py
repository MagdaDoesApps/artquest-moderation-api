from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

app = FastAPI()

# Load YOLO model once
model = YOLO("yolov8n.pt")  # Tiny, efficient model

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

        # Try opening image with Pillow
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # Run YOLO prediction
        results = model.predict(img, verbose=False)

        # Extract class names (labels)
        labels = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            labels.append(model.names[class_id])

        # Define which classes might be unsafe (can tweak)
        unsafe_keywords = ["knife", "knife", "gun", "weapon", "blood", "nude", "person"]

        # Check if any unsafe label is detected
        unsafe = any(any(k in label.lower() for k in unsafe_keywords) for label in labels)

        return {
            "safe": not unsafe,
            "labels": labels,
            "unsafe_labels": [l for l in labels if any(k in l.lower() for k in unsafe_keywords)]
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)



