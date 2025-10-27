from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests
import os

app = FastAPI()

# Load YOLOv8 model (lightweight)
weapon_model = YOLO("yolov8n.pt")

def is_nsfw_image(image: Image.Image) -> float:
    """Return a score between 0 and 1 for NSFW likelihood (rough skin-tone heuristic)."""
    img = image.convert("RGB").resize((128, 128))
    pixels = img.load()
    total, skin_like = 0, 0
    for x in range(img.width):
        for y in range(img.height):
            r, g, b = pixels[x, y]
            total += 1
            if r > 95 and g > 40 and b > 20 and max(r, g, b)-min(r, g, b) > 15 and r > g and r > b:
                skin_like += 1
    return skin_like / total

class ImageInput(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"message": "✅ ArtQuest Moderation API is online!"}

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # YOLO detection for weapons/gore
    results = weapon_model.predict(img, verbose=False)
    labels = [weapon_model.names[int(b.cls)] for b in results[0].boxes]

    # Scoring
    weapon_score = 1.0 if any(k in l.lower() for l in labels for k in ["knife", "gun", "weapon"]) else 0.0
    gore_score = 1.0 if any(k in l.lower() for l in labels for k in ["blood"]) else 0.0
    nsfw_score = is_nsfw_image(img)
    violence_score = max(weapon_score, gore_score)  # rough approximation

    safe = nsfw_score < 0.35 and weapon_score == 0.0 and gore_score == 0.0

    return {
        "Safe": safe,
        "NSFWScore": nsfw_score,
        "ViolenceScore": violence_score,
        "WeaponScore": weapon_score,
        "GoreScore": gore_score
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
