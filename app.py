import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ====== 動態模型路徑修正 ======
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(SRC_DIR, "models")

print("📌 BASE_DIR =", BASE_DIR)
print("📌 MODEL_DIR =", MODEL_DIR)

# ====== 匯入自製模組（注意：都在 src/v3 下） ======
from src.face_detector import load_detector, detect_faces
from src.embedding import load_embedder, get_embedding
from src.classifier_cosine import cosine_predict
from src.classifier_svm import load_svm, svm_predict
from src.classifier_knn import load_knn, knn_predict

app = FastAPI()


# ====== numpy safe converter ======
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


# ====== 載入所有模型 ======
print("📌 Loading models...")

detector = load_detector()          # buffalo_l detector
embedder = load_embedder()          # arcface embedder

svm_model = load_svm(os.path.join(MODEL_DIR, "svm.pkl"))
knn_model = load_knn(os.path.join(MODEL_DIR, "knn.pkl"))

print("✅ All models loaded successfully!")


# ====== Request Schema ======
class PredictRequest(BaseModel):
    image_base64: str
    record_id: str


# ====== API Home ======
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "TSUKINEKO face API running"
    }


# ====== 主辨識 API ======
@app.post("/predict")
def predict(req: PredictRequest):

    # 1. base64 → raw bytes
    img_data = base64.b64decode(req.image_base64)

    # 2. raw bytes → numpy uint8 array
    img_array = np.frombuffer(img_data, np.uint8)

    # 3. numpy array → OpenCV image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Image decode failed"}

    # 4. detect face
    faces = detect_faces(detector, img_array)  # ⬅️ 注意：丟 array，不是丟 img（你的 detect_faces 會自己 decode）

    if len(faces) == 0:
        return {"status": "no_face", "record_id": req.record_id}

    # 5. 用第一張臉抽 embedding
    face = faces[0]
    emb = get_embedding(embedder, img, face["bbox"])

    # 6. Ensemble
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
