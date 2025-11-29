import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from src.face_detector import load_detector, detect_faces
from src.embedding import load_embedder, get_embedding
from src.classifier_cosine import cosine_predict
from src.classifier_svm import load_svm, svm_predict
from src.classifier_knn import load_knn, knn_predict

app = FastAPI()

def safe_convert(o):
    if isinstance(o, (np.int64, np.int32, np.uint8)):
        return int(o)
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, dict):
        return {k: safe_convert(v) for k, v in o.items()}
    if isinstance(o, list):
        return [safe_convert(v) for v in o]
    return o
print("📌 Loading models...")

detector = load_detector()
embedder = load_embedder()
svm_model = load_svm()
knn_model = load_knn()

print("✅ All models loaded successfully!")
class PredictRequest(BaseModel):
    image_base64: str
    record_id: str
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "TSUKINEKO face API running"
    }
@app.post("/predict")
def predict(req: PredictRequest):

    img_data = base64.b64decode(req.image_base64)
    np_img = np.frombuffer(img_data, np.uint8)
    faces = detect_faces(detector, np_img)

    if len(faces) == 0:
        return {"status": "no_face", "record_id": req.record_id}

    face = faces[0]
    emb = get_embedding(embedder, np_img, face["bbox"])

    # Ensemble
    cos_label, cos_score = cosine_predict(emb)
    svm_label, svm_score = svm_predict(svm_model, emb)
    knn_label, knn_score = knn_predict(knn_model, emb)

    votes = [cos_label, svm_label, knn_label]
    final_label = max(set(votes), key=votes.count)

    return safe_convert({
        "status": "ok",
        "record_id": req.record_id,
        "final_pred": final_label,
        "details": {
            "cosine": {"pred": cos_label, "score": cos_score},
            "svm": {"pred": svm_label, "score": svm_score},
            "knn": {"pred": knn_label, "score": knn_score},
        }
    })
