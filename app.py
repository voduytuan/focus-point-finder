from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
import insightface
import numpy as np
import cv2
import urllib.request
from PIL import Image, ImageDraw
import io

app = FastAPI()

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

def download_image(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            image_data = resp.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise RuntimeError(f"Failed to download or parse image: {e}")

def get_saliency_point(img):
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(img)
        if success:
            saliency_map = (saliency_map * 255).astype("uint8")
            _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
            return max_loc, saliency_map
    except Exception as e:
        print("⚠️ Saliency error:", e)
    return None, None

def get_focus_point(img, faces):
    if faces:
        x = int(np.mean([(f.bbox[0] + f.bbox[2]) / 2 for f in faces]))
        y = int(np.mean([(f.bbox[1] + f.bbox[3]) / 2 for f in faces]))
        print("✅ Using face center")
        return (x, y), None

    print("⚠️ No face found, using saliency fallback")
    point, _ = get_saliency_point(img)
    if point:
        print("✅ Saliency returned:", point)
        return point, None

    h, w = img.shape[:2]
    print("⚠️ Final fallback to center")
    return (w // 2, h // 2), None

@app.get("/focus-point")
def focus_point(image_url: str = Query(..., alias="url")):
    try:
        img = download_image(image_url)
        faces = model.get(img)
        (x, y), _ = get_focus_point(img, faces)
        return {"x": x, "y": y}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/debug/faces")
def debug_faces(image_url: str = Query(..., alias="url")):
    try:
        img = download_image(image_url)
        faces = model.get(img)
        return [{"bbox": list(map(int, f.bbox))} for f in faces]
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/debug/image-with-boxes")
def debug_image_with_boxes(image_url: str = Query(..., alias="url")):
    try:
        img = download_image(image_url)
        faces = model.get(img)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        for f in faces:
            box = list(map(int, f.bbox))
            draw.rectangle(box, outline="red", width=3)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/debug/image-with-focus")
def debug_image_with_focus(image_url: str = Query(..., alias="url")):
    try:
        img = download_image(image_url)
        faces = model.get(img)
        (x, y), _ = get_focus_point(img, faces)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/debug/saliency-map")
def debug_saliency_map(image_url: str = Query(..., alias="url")):
    try:
        img = download_image(image_url)
        point, saliency_map = get_saliency_point(img)
        if saliency_map is None:
            raise RuntimeError("Could not compute saliency map.")
        sal_pil = Image.fromarray(saliency_map)
        buf = io.BytesIO()
        sal_pil.save(buf, format="JPEG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
