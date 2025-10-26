from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os, torch

app = FastAPI()

model = YOLO("yolov8n.pt")  # tiny model, low memory

class ImageInput(BaseModel):
    image_url: str

@app.post("/moderate")
async def moderate_image(data: ImageInput):
    try:
        response = requests.get(data.image_url)
        img = Image.open(BytesIO(response.content))
        results = model.predict(img)
        # Your own simple logic, e.g. analyze class names
        labels = [c.name for c in results[0].boxes.cls]
        return {"labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
