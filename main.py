from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests
import os
import onnxruntime_lite as ort
import numpy as np

app = FastAPI()

# ---- Load models ----
print("Loading models...")

# Load YOLO for weapons/gore
weapon_model = YOLO("yolov8n.pt")

# Load lightweight NSFW model (.onnx)
nsfw_session = ort.InferenceSession("nsfw_mobilenet.onnx", providers=["CPUExecutionProvider"])

@app.get("/")
def root():
    return {"message": "✅ ArtQuest Moderation API is online!"}


class ImageInput(BaseModel):
    image_url: str


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize and normalize for ONNX NSFW model."""
    img = img.resize((224, 224)).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
    return arr


@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        # --- Download image ---
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # --- YOLO detection ---
        results = weapon_model.predict(img, verbose=False)
        labels = [weapon_model.names[int(box.cls)] for box in results[0].boxes]

        # --- NSFW detection ---
        img_array = preprocess(img)
        nsfw_inputs = {nsfw_session.get_inputs()[0].name: img_array}
        preds = nsfw_session.run(None, nsfw_inputs)[0][0]
        nsfw_score = float(preds[1])  # Assume index 1 is NSFW class probability

        # --- Decide ---
        unsafe_keywords = ["knife", "gun", "weapon", "blood"]
        has_weapon = any(k in label.lower() for label in labels for k in unsafe_keywords)
        unsafe = nsfw_score > 0.3 or has_weapon

        return {
            "safe": not unsafe,
            "nsfw_score": round(nsfw_score, 3),
            "labels": labels,
            "unsafe_reason": {
                "nsfw": nsfw_score > 0.3,
                "weapon": has_weapon
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

