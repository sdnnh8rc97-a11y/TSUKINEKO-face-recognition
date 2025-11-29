import cv2
from insightface.app import FaceAnalysis
import numpy as np

def load_detector():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app

def detect_faces(app, image_bytes):
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    faces = app.get(img)
    result = []
    for f in faces:
        result.append({
            "bbox": f.bbox.astype(int).tolist(),
            "det_score": float(f.det_score)
        })
    return result
