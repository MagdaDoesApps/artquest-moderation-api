from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests
import os

app = FastAPI()

# ---- Load lightweight YOLOv8 models ----
# 1️⃣ General object detector (weapons, knives, guns)
weapon_model = YOLO("yolov8n.pt")

# 2️⃣ Lightweight NSFW classifier (custom threshold using skin detection)
#     Instead of a heavy CNN, we'll just do a color-based approximation.
def is_nsfw_image(image: Image.Image) -> bool:
    # Very rough "skin-tone ratio" heuristic
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
    return ratio > 0.35  # tweak threshold for strictness


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

    # ---- YOLO weapon/gore detection ----
    results = weapon_model.predict(img, verbose=False)
    labels = [weapon_model.names[int(b.cls)] for b in results[0].boxes]

    unsafe_keywords = ["knife", "gun", "weapon", "blood"]
    weapon_detected = any(k in l.lower() for l in labels for k in unsafe_keywords)

    # ---- Color-based NSFW check ----
    nsfw_detected = is_nsfw_image(img)

    unsafe = weapon_detected or nsfw_detected
    unsafe_reasons = []
    if weapon_detected:
        unsafe_reasons.append("weapon")
    if nsfw_detected:
        unsafe_reasons.append("nsfw/sexual")

    return {
        "safe": not unsafe,
        "unsafe_reasons": unsafe_reasons,
        "labels": labels,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
