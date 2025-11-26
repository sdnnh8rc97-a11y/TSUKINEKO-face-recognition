
import os
import json
import pickle
import numpy as np
import cv2
from numpy.linalg import norm
from insightface.app import FaceAnalysis

MODEL_DIR = "/content/face_system/models"

COSINE_WEIGHT = 0.5
KNN_WEIGHT = 0.3
SVM_WEIGHT = 0.2

MIN_FACE_SIZE = 40  # 小於這個框的臉自動 Unknown


# ============================
# Safe imread（支援中文路徑）
# ============================
def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


# ============================
# 載入模型
# ============================
def load_models():
    with open(f"{MODEL_DIR}/knn.pkl", "rb") as f:
        knn = pickle.load(f)

    with open(f"{MODEL_DIR}/svm.pkl", "rb") as f:
        svm = pickle.load(f)

    with open(f"{MODEL_DIR}/centers.pkl", "rb") as f:
        centers = pickle.load(f)

    with open(f"{MODEL_DIR}/label_map.json", "r") as f:
        label_map = json.load(f)

    with open(f"{MODEL_DIR}/threshold.json", "r") as f:
        thr = json.load(f)["cosine_threshold"]

    inv_label = {v: k for k, v in label_map.items()}

    return knn, svm, centers, label_map, inv_label, thr


# ============================
# Cosine similarity
# ============================
def compute_cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# ============================
# 單臉 Ensemble 推論
# ============================
def classify_embedding(emb, knn, svm, centers, inv_label, cosine_thr):
    # --- 1. cosine ---
    cos_scores = {}
    for person, center in centers.items():
        cos_scores[person] = compute_cosine(emb, center)

    cosine_pred = max(cos_scores, key=cos_scores.get)
    cosine_conf = cos_scores[cosine_pred]

    if cosine_conf < cosine_thr:
        cosine_pred = "Unknown"

    # --- 2. KNN ---
    knn_pred_raw = knn.predict([emb])[0]
    # 計算 knn confidence：k 鄰居有多少同 label
    dist, idx = knn.kneighbors([emb], n_neighbors=3, return_distance=True)
    labels = knn.classes_
    neighbor_labels = [knn._y[i] for i in idx[0]]
    knn_conf = neighbor_labels.count(knn_pred_raw) / 3.0

    knn_pred = knn_pred_raw if knn_conf >= 0.34 else "Unknown"

    # --- 3. SVM ---
    svm_probs = svm.predict_proba([emb])[0]
    max_idx = np.argmax(svm_probs)
    svm_pred_raw = svm.classes_[max_idx]
    svm_conf = svm_probs[max_idx]

    svm_pred = svm_pred_raw if svm_conf >= 0.40 else "Unknown"

    # ============================
    # Weighted Ensemble
    # ============================
    score_map = {}

    for name in centers.keys():
        score_map[name] = (
            (cos_scores.get(name, 0) * COSINE_WEIGHT) +
            (knn_conf if knn_pred_raw == name else 0) * KNN_WEIGHT +
            (svm_conf if svm_pred_raw == name else 0) * SVM_WEIGHT
        )

    # Unknown Score
    unknown_score = (
        (1 - cosine_conf) * COSINE_WEIGHT +
        (1 - knn_conf) * KNN_WEIGHT +
        (1 - svm_conf) * SVM_WEIGHT
    )

    final_pred = max(score_map, key=score_map.get)
    final_score =

