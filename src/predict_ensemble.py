# ===============================================================
# ensemble_predict_clean_C_with_knn_distance.py
# (Cosine + SVM + KNN Distance) — 完整修正版
# ===============================================================

import os
import cv2
import json
import pickle
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
from google.colab.patches import cv2_imshow

# ---------------------------------------------------------------
# JSON 安全轉換器
# ---------------------------------------------------------------
def safe_convert(o):
    import numpy as np
    if isinstance(o, (np.int64, np.int32, np.uint8)):
        return int(o)
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, dict):
        return {k: safe_convert(v) for k, v in o.items()}
    if isinstance(o, list):
        return [safe_convert(v) for v in o]
    return o

def safe_json(data):
    import json
    return json.dumps(safe_convert(data), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------
# 全域設定
# ---------------------------------------------------------------
MODEL_DIR = "/content/drive/MyDrive/face_DataSet/models"
FONT_PATH = "/content/NotoSansCJK-Regular.otf"
MIN_FACE_SIZE = 55

COSINE_WEIGHT = 0.25
SVM_WEIGHT = 0.50
KNN_WEIGHT = 0.25 
FINAL_CONF_THRESHOLD = 0.35

# ---------------------------------------------------------------
# 工具
# ---------------------------------------------------------------
def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def normalize_embedding(face):
    return face.normed_embedding.astype("float32")

# ---------------------------------------------------------------
# 載入模型
# ---------------------------------------------------------------
def initialize_system():
    print("🔧 載入模型中...")

    with open(f"{MODEL_DIR}/knn.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(f"{MODEL_DIR}/svm.pkl", "rb") as f:
        svm = pickle.load(f)
    with open(f"{MODEL_DIR}/centers.pkl", "rb") as f:
        centers = pickle.load(f)
    with open(f"{MODEL_DIR}/label_map.json", "r") as f:
        label_map = json.load(f)
    with open(f"{MODEL_DIR}/threshold.json", "r") as f:
        cosine_thr = json.load(f)["balanced"]
        # 放寬 cosine 門檻（建議 ↓）
        cosine_thr = max(0.50, cosine_thr - 0.10)


    inv_label = {v: k for k, v in label_map.items()}
    y_true = np.load(f"{MODEL_DIR}/y.npy")

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(1280, 1280))

    print("✅ Initialization Done!")
    return knn, svm, centers, label_map, inv_label, cosine_thr, y_true, app

# ---------------------------------------------------------------
# KNN 距離模式
# ---------------------------------------------------------------
def knn_predict_distance(knn, emb, inv_label, thr=0.50):
    dists, idx = knn.kneighbors([emb], n_neighbors=1, return_distance=True)
    best_dist = float(dists[0][0])
    best_idx = int(knn.predict([emb])[0])
    best_label = inv_label.get(best_idx, "Unknown")

    if best_dist <= thr:
        return best_label, best_dist
    return "Unknown", best_dist

# ---------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------
def classify_embedding(emb, knn, svm, centers, inv_label, cosine_thr, y_true):

    results = {}

    # --- 1. Cosine ---
    cos_scores = {p: cosine_similarity(emb, c) for p, c in centers.items()}
    best_cos = max(cos_scores, key=cos_scores.get)
    cos_conf = cos_scores[best_cos]

    cos_pred = best_cos if cos_conf >= cosine_thr else "Unknown"
    if cos_pred != "Unknown":
        results[cos_pred] = results.get(cos_pred, 0) + cos_conf * COSINE_WEIGHT

    # --- 2. SVM ---
    svm_probs = svm.predict_proba([emb])[0]
    svm_idx = np.argmax(svm_probs)

    # 這是 SVM 的 class index（數字）
    svm_class_index = svm.classes_[svm_idx]

    # 使用 inv_label 把數字轉回人名
    svm_pred_raw = inv_label.get(svm_class_index, "Unknown")

    svm_conf_raw = float(svm_probs[svm_idx])

    if svm_conf_raw >= 0.40:
        results[svm_pred_raw] = results.get(svm_pred_raw, 0) + svm_conf_raw * SVM_WEIGHT

    
    # --- 3. KNN ---
    knn_pred_raw, knn_dist = knn_predict_distance(knn, emb, inv_label, thr=0.50)

    if knn_pred_raw != "Unknown":
        knn_conf = max(0, 1 - knn_dist)
        results[knn_pred_raw] = results.get(knn_pred_raw, 0) + knn_conf * KNN_WEIGHT
    else:
        knn_conf = 0.0

    # --- Final ---
    if not results:
        return {
            "final_pred": "Unknown",
            "final_conf": 0.0,
            "cosine_pred": cos_pred,
            "cosine_conf": float(cos_conf),
            "svm_pred": svm_pred_raw,
            "svm_conf": float(svm_conf_raw),
            "knn_pred": knn_pred_raw,
            "knn_conf": float(knn_conf),
        }

    final_pred = max(results, key=results.get)
    final_conf = results[final_pred]

    if final_conf < FINAL_CONF_THRESHOLD:
        final_pred = "Unknown"

    return {
        "final_pred": final_pred,
        "final_conf": float(final_conf),
        "cosine_pred": cos_pred,
        "cosine_conf": float(cos_conf),
        "svm_pred": svm_pred_raw,
        "svm_conf": float(svm_conf_raw),
        "knn_pred": knn_pred_raw,
        "knn_conf": float(knn_conf),
    }

# ---------------------------------------------------------------
# 團體照
# ---------------------------------------------------------------
def recognize_image(img_path, app, knn, svm, centers, inv_label, cosine_thr, y_true):

    img = imread_safe(img_path)
    draw_img = img.copy()
    faces = app.get(img)
    results = []

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        w = x2 - x1

        if w < MIN_FACE_SIZE:
            results.append({
                "bbox": (x1, y1, x2, y2),
                "final_pred": "Unknown (Small)",
                "details": {}
            })
            continue

        emb = normalize_embedding(f)
        detail = classify_embedding(emb, knn, svm, centers, inv_label, cosine_thr, y_true)

        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # ⚠⚠⚠ 修正後（本來就應該用 detail 的 final_pred）
        results.append({
            "bbox": (x1, y1, x2, y2),
            "final_pred": detail["final_pred"],   # ← 修正！
            "details": detail
        })

    return results, draw_img

# ---------------------------------------------------------------
# PIL 畫中文字
# ---------------------------------------------------------------

def draw_results_pil(image, detections, font_path="/content/NotoSansCJK-Regular.otf"):
    # OpenCV → PIL
    
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = ImageFont.truetype(font_path, 20)  # 字體大小可以調

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name = det["final_pred"]

        # 畫框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=4)

        # 畫中文名字
        draw.text((x1, y1 - 45), name, font=font, fill=(0, 255, 0))

    # PIL → OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":

    knn, svm, centers, label_map, inv_label, cosine_thr, y_true, app = initialize_system()

    INPUT_IMAGE = "/content/drive/MyDrive/test_faces/保全group測試/IMG_3744.JPG"
    
    results, rawimg = recognize_image(
        INPUT_IMAGE, app, knn, svm, centers, inv_label, cosine_thr, y_true
    )
    img_path = "/content/drive/MyDrive/test_faces/保全group測試/IMG_3744.JPG"
    img = imread_safe(img_path)
    print(safe_json(results))
    

    output = draw_results_pil(img, results)
    cv2_imshow(output)
   
