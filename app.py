from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from PIL import Image, ImageDraw, UnidentifiedImageError
import numpy as np
import io
import cv2
import insightface

app = FastAPI()
face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)

class FocusPoint(BaseModel):
    x: int
    y: int

def get_image_from_url(image_url: str):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot fetch image from URL.")
    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="URL does not point to a valid image.")
    return image

def get_focus_point_from_image(np_image: np.ndarray) -> tuple[int, int]:
    faces = face_model.get(np_image)
    if faces:
        sorted_faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[:3]
        centers = [((f.bbox[0]+f.bbox[2])//2, (f.bbox[1]+f.bbox[3])//2) for f in sorted_faces]
        x = sum([c[0] for c in centers]) // len(centers)
        y = sum([c[1] for c in centers]) // len(centers)
        return int(x), int(y)
    else:
        image_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image_bgr)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(saliencyMap)
        return int(maxLoc[0]), int(maxLoc[1])

@app.get("/focus-point", response_model=FocusPoint)
def get_focus_point(image_url: str = Query(...)):
    image = get_image_from_url(image_url)
    np_image = np.array(image)
    x, y = get_focus_point_from_image(np_image)
    return FocusPoint(x=x, y=y)

@app.get("/debug/face-boxes")
def get_face_boxes(image_url: str = Query(...)):
    image = get_image_from_url(image_url)
    np_image = np.array(image)
    faces = face_model.get(np_image)
    boxes = []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return JSONResponse(content=boxes)

@app.get("/debug/image-with-boxes")
def image_with_boxes(image_url: str = Query(...)):
    image = get_image_from_url(image_url)
    draw = ImageDraw.Draw(image)
    np_image = np.array(image)
    faces = face_model.get(np_image)
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return Response(content=output.getvalue(), media_type="image/jpeg")

@app.get("/debug/image-with-focus")
def image_with_focus(image_url: str = Query(...)):
    image = get_image_from_url(image_url)
    np_image = np.array(image)
    draw = ImageDraw.Draw(image)
    x, y = get_focus_point_from_image(np_image)
    draw.ellipse([(x-5, y-5), (x+5, y+5)], fill="red")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return Response(content=output.getvalue(), media_type="image/jpeg")
