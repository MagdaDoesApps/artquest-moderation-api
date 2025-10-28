from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

app = FastAPI()

# Load YOLOv8n model (realistic objects)
model = YOLO("yolov8n.pt")  # tiny, fast

class ImageInput(BaseModel):
    image_url: str


def is_nsfw_image(image: Image.Image) -> bool:
    """Simple skin-tone ratio detector (rough NSFW filter)."""
    img = image.convert("RGB").resize((128, 128))
    pixels = img.load()
    total, skin_like = 0, 0
    for x in range(img.width):
        for y in range(img.height):
            r, g, b = pixels[x, y]
            total += 1
            if r > 95 and g > 40 and b > 20 and max(r, g, b) - min(r, g, b) > 15 and r > g and r > b:
                skin_like += 1
    ratio = skin_like / total
    return ratio > 0.35


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

    # --- YOLOv8 weapon/gore detection ---
    results = model.predict(img, verbose=False)
    labels = [model.names[int(b.cls)] for b in results[0].boxes]

    # Define unsafe real-world categories
    unsafe_keywords = ["knife", "gun", "weapon", "blood"]

    # Check for real-world weapon-like detections
    weapon_detected = any(k in l.lower() for l in labels for k in unsafe_keywords)

    # --- NSFW heuristic check ---
    nsfw_detected = is_nsfw_image(img)

    # --- Final moderation logic ---
    unsafe_reasons = []
    if weapon_detected:
        unsafe_reasons.append("weapon/violence")
    if nsfw_detected:
        unsafe_reasons.append("nsfw/sexual")

    return {
        "safe": not (weapon_detected or nsfw_detected),
        "unsafe_reasons": unsafe_reasons,
        "labels": labels,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
