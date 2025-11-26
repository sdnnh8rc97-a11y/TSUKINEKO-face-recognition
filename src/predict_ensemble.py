import os
import json
import joblib
import numpy as np

from face_embedder import FaceEmbedder
from face_detector import FaceDetector

MODEL_DIR = "./models"
UNKNOWN_THRESHOLD = 0.45  # 建議先用 0.45，之後再微調

# Load models
knn = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
svm = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
label_map = json.load(open(os.path.join(MODEL_DIR, "label_map.json")))
centers = json.load(open(os.path.join(MODEL_DIR, "centers.json")))

# 工具
embedder = FaceEmbedder()
detector = FaceDetector()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# Ensemble Voting
# -----------------------------
def ensemble_predict(emb):
    # 1️⃣ Center-based
    center_scores = {p: cosine(emb, np.array(vec)) for p, vec in centers.items()}
    center_best = max(center_scores, key=center_scores.get)
    center_conf = center_scores[center_best]

    # 2️⃣ KNN
    knn_prob = knn.predict_proba([emb])[0]
    knn_idx = np.argmax(knn_prob)
    knn_name = label_map[str(knn_idx)]
    knn_conf = knn_prob[knn_idx]

    # 3️⃣ SVM
    svm_prob = svm.predict_proba([emb])[0]
    svm_idx = np.argmax(svm_prob)
    svm_name = label_map[str(svm_idx)]
    svm_conf = svm_prob[svm_idx]

    # -------------------------
    # Ensemble 合併
    # 加權可調整（目前：center 0.3 / knn 0.3 / svm 0.4）
    # -------------------------
    final_scores = {}

    for person in label_map.values():
        idx = list(label_map.values()).index(person)

        final_scores[person] = (
            0.30 * center_scores.get(person, 0) +
            0.30 * knn_prob[idx] +
            0.40 * svm_prob[idx]
        )

    best_person = max(final_scores, key=final_scores.get)
    best_score = final_scores[best_person]

    # Unknown 判斷
    if best_score < UNKNOWN_THRESHOLD:
        return "Unknown", float(best_score)

    return best_person, float(best_score)


# -----------------------------
# 實際推論流程
# -----------------------------
def predict_image(image_path):
    faces = detector.detect(image_path)
    results = []

    for face in faces:
        emb = embedder.get_embedding(face["crop"])
        if emb is None:
            continue

        name, conf = ensemble_predict(emb)

        results.append({
            "name": name,
            "confidence": round(conf, 3),
            "bbox": face["bbox"]
        })

    return results
