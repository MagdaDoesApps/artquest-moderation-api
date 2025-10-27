from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO           # For NSFW detection
import torch                           # For YOLOv5 (weapons)
from PIL import Image
import requests
from io import BytesIO
import os

app = FastAPI()

# --- Load models once ---
nsfw_model = YOLO("yolov8n.pt")              # NSFW detection
weapons_model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="best.pt", force_reload=True
)                                             # Weapons detection

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

        # Open image with Pillow
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # --- NSFW Detection ---
        nsfw_results = nsfw_model.predict(img, verbose=False)
        nsfw_labels = [
            nsfw_model.names[int(box.cls)] for box in nsfw_results[0].boxes
        ]
        nsfw_keywords = ["nude", "person", "sex"]
        nsfw_flag = any(any(k in label.lower() for k in nsfw_keywords) for label in nsfw_labels)

        # --- Weapons Detection ---
        # Convert PIL image to numpy array
        img_np = np.array(img)
        weapons_results = weapons_model(img_np)
        weapons_labels = [weapons_results.names[int(cls)] for cls in weapons_results.xyxy[0][:, 5]]
        weapons_keywords = ["knife", "gun", "weapon"]
        weapons_flag = any(any(k in label.lower() for k in weapons_keywords) for label in weapons_labels)

        # Overall unsafe flag
        unsafe = nsfw_flag or weapons_flag

        return {
            "safe": not unsafe,
            "nsfw_labels": nsfw_labels,
            "weapons_labels": weapons_labels,
            "unsafe_labels": nsfw_labels + weapons_labels,
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
