
import os
import json
import pickle
import numpy as np
import cv2
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# ======================================================
# 配置
# ======================================================
MODEL_DIR = "/content/face_system/models"
MIN_FACE_SIZE = 40  # 小於這個框 → 自動 Unknown

# Ensemble 權重（可微調）
COSINE_WEIGHT = 0.5
KNN_WEIGHT = 0.3
SVM_WEIGHT = 0.2


# ======================================================
# Safe imread（支援中文路徑）
# ======================================================
def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


# ======================================================
# Cosine similarity
# ======================================================
def compute_cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# ======================================================
# 載入模型
# ======================================================
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
        cosine_thr = json.load(f)["cosine_threshold"]

    inv_label = {v: k for k, v in label_map.items()}
    return knn, svm, centers, label_map, inv_label, cosine_thr


# ======================================================
# Ensemble 推論（單一 embedding）
# ======================================================
def classify_embedding(emb, knn, svm, centers, inv_label, cosine_thr):
    # -------------------------
    # 1. COSINE
    # -------------------------
    cos_scores = {p: compute_cosine(emb, c) for p, c in centers.items()}
    raw_cos_pred = max(cos_scores, key=cos_scores.get)
    raw_cos_conf = cos_scores[raw_cos_pred]

    if raw_cos_conf >= cosine_thr:
        cosine_pred = raw_cos_pred
        cosine_conf = raw_cos_conf
    else:
        cosine_pred = "Unknown"
        cosine_conf = 0.0

    # -------------------------
    # 2. KNN
    # -------------------------
    knn_pred_raw = knn.predict([emb])[0]

    dist, idx = knn.kneighbors([emb], n_neighbors=3, return_distance=True)
    neighbor_labels = [knn._y[i] for i in idx[0]]
    raw_knn_conf = neighbor_labels.count(knn_pred_raw) / 3.0

    if raw_knn_conf >= 0.34:
        knn_pred = knn_pred_raw
        knn_conf = raw_knn_conf
    else:
        knn_pred = "Unknown"
        knn_conf = 0.0

    # -------------------------
    # 3. SVM
    # -------------------------
    svm_probs = svm.predict_proba([emb])[0]
    max_idx = np.argmax(svm_probs)
    svm_pred_raw = svm.classes_[max_idx]
    raw_svm_conf = float(svm_probs[max_idx])

    if raw_svm_conf >= 0.40:
        svm_pred = svm_pred_raw
        svm_conf = raw_svm_conf
    else:
        svm_pred = "Unknown"
        svm_conf = 0.0

    # =====================================================
    # 4. Majority Voting
    # =====================================================
    votes = []
    if cosine_pred != "Unknown":
        votes.append(cosine_pred)
    if knn_pred != "Unknown":
        votes.append(knn_pred)
    if svm_pred != "Unknown":
        votes.append(svm_pred)

    if len(votes) == 0:
        final_pred = "Unknown"
    else:
        final_pred = max(set(votes), key=votes.count)

        # 最終安全門：cosine 必須支持
        if cosine_conf < cosine_thr:
            final_pred = "Unknown"

    return {
        "cosine_pred": cosine_pred,
        "cosine_conf": float(cosine_conf),
        "knn_pred": knn_pred,
        "knn_conf": float(knn_conf),
        "svm_pred": svm_pred,
        "svm_conf": float(svm_conf),
        "votes": votes,
        "final_pred": final_pred,
    }


# ======================================================
# 圖片辨識（支援團體照）
# ======================================================
def recognize_image(image_path, knn, svm, centers, inv_label, cosine_thr):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)

    img = imread_safe(image_path)
    if img is None:
        raise ValueError("❌ 圖片讀取失敗")

    faces = app.get(img)
    results = []

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # 太小的臉不辨識
        if (x2 - x1) < MIN_FACE_SIZE:
            results.append({
                "pred": "Unknown",
                "reason": "face too small",
                "box": (x1, y1, x2, y2)
            })
            continue

        emb = face.normed_embedding
        info = classify_embedding(emb, knn, svm, centers, inv_label, cosine_thr)

        results.append({
            "bbox": (x1, y1, x2, y2),
            "final_pred": info["final_pred"],
            "details": info
        })

    return results


# ======================================================
# 主程式 CLI
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()

    knn, svm, centers, label_map, inv_label, cosine_thr = load_models()

    print(f"🚀 Threshold = {cosine_thr}")
    print("📂 Loading image:", args.img)

    output = recognize_image(args.img, knn, svm, centers, inv_label, cosine_thr)

    print("\n=== 辨識結果 ===")
    for i, f in enumerate(output):
        print(f"\n臉 #{i+1}")
        print("位置：", f["bbox"])
        print("辨識：", f["final_pred"])
        print("詳細：", f["details"])
