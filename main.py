from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nsfw_detector import predict
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests
import os

app = FastAPI()

# Load lightweight NSFW model
nsfw_model = predict.load_model("nsfw_model.h5")

# Load tiny YOLOv8 model for object detection (guns, knives, etc.)
weapon_model = YOLO("yolov8n.pt")

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
        img_bytes = BytesIO(response.content)
        img = Image.open(img_bytes).convert("RGB")

        # NSFW check
        nsfw_result = predict.classify(nsfw_model, data.image_url)
        nsfw_scores = list(nsfw_result.values())[0]

        nsfw_score = nsfw_scores.get("sexy", 0) + nsfw_scores.get("porn", 0) + nsfw_scores.get("hentai", 0)

        # Weapon detection (YOLO)
        results = weapon_model.predict(img, verbose=False)
        labels = [weapon_model.names[int(box.cls)] for box in results[0].boxes]
        unsafe_weapons = any(label.lower() in ["knife", "gun", "weapon"] for label in labels)

        # Final decision
        unsafe = nsfw_score > 0.3 or unsafe_weapons
        return {
            "safe": not unsafe,
            "nsfw_score": round(nsfw_score, 3),
            "labels": labels,
            "unsafe_reason": {
                "nsfw": nsfw_score > 0.3,
                "weapon": unsafe_weapons
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
