import io
import os
import shutil
import uuid
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from scripts.inference import predict, preprocess

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG or JPG images are supported.")
    
    if file.size and file.size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    os.makedirs("temp", exist_ok=True)

    input_path = os.path.join("temp", f"input_{uuid.uuid4()}.png")

    try:
        # Save uploaded image
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess and predict
        with open(input_path, "rb") as img_buffer:
            image_tensor = preprocess(img_buffer)
        pred_mask = predict(image_tensor)

        # Save predicted mask into memory
        mask_img = Image.fromarray((pred_mask * 85).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)

    except Exception as e:
        logging.exception("Prediction or saving failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return StreamingResponse(buf, media_type="image/png", headers={"Content-Disposition": "attachment; filename=predicted_mask.png"})
