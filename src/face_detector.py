import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_detector():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app

def detect_faces(app, image_bytes):
    # raw bytes → uint8 array
    img_array = np.frombuffer(image_bytes, np.uint8)

    # uint8 array → OpenCV image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return []  # 無法解碼

    # insightface 偵測
    faces = app.get(img)

    result = []
    for f in faces:
        result.append({
            "bbox": f.bbox.astype(int).tolist(),
            "det_score": float(f.det_score)
        })

    return result
