# ===============================================================
#  三合一臉辨系統（高速版 + Anti-Drift Auto-Train + 團體照分類）
#  作者：まさき專用（V3.1 安全版） + Cloud Run API 支援
# ===============================================================

import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

# ---------------------------------------------------------------
# 設定路徑（Cloud Run 不會跑 retrain，所以這些會自動失效）
# ---------------------------------------------------------------
RAW_ROOT = "/content/drive/MyDrive/face_DataSet/face_raw"
CACHE_ROOT = "/content/drive/MyDrive/face_DataSet/face_emb_cache"
CLASSIFY_SAVE = "/content/drive/MyDrive/face_DataSet/face_clean_group"
GROUP_PHOTO = "/content/drive/MyDrive/test_faces/保全group測試/19534.jpg"

# ---------------------------------------------------------------
# 初始化 InsightFace
# ---------------------------------------------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ===============================================================
#  STEP 1 — 建立 embedding cache
# ===============================================================
def build_cache():
    os.makedirs(CACHE_ROOT, exist_ok=True)

    persons = sorted([
        p for p in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, p))
    ])

    for person in persons:
        raw_dir = os.path.join(RAW_ROOT, person)
        cache_dir = os.path.join(CACHE_ROOT, person)
        os.makedirs(cache_dir, exist_ok=True)

        photos = [
            f for f in os.listdir(raw_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        for img_name in photos:
            raw_path = os.path.join(raw_dir, img_name)
            cache_path = os.path.join(cache_dir, img_name + ".npy")

            if os.path.exists(cache_path):
                continue

            img = cv2.imread(raw_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            emb = faces[0].normed_embedding
            np.save(cache_path, emb)

# ===============================================================
#  STEP 2 — 從 cache 建立資料庫（平均 embedding）
# ===============================================================
def load_database():
    db = {}

    for person in os.listdir(CACHE_ROOT):
        p_dir = os.path.join(CACHE_ROOT, person)
        if not os.path.isdir(p_dir):
            continue

        embs = []
        for f in os.listdir(p_dir):
            if f.endswith(".npy"):
                emb = np.load(os.path.join(p_dir, f))
                embs.append(emb)

        if len(embs) > 0:
            db[person] = np.mean(embs, axis=0)

    return db

# ===============================================================
# STEP 2.1 — Cluster Stats（中心＋標準差）
# ===============================================================
def build_db_stats():
    stats = {}

    for person in os.listdir(CACHE_ROOT):
        p_dir = os.path.join(CACHE_ROOT, person)
        if not os.path.isdir(p_dir):
            continue

        embs = []
        for f in os.listdir(p_dir):
            if f.endswith(".npy"):
                embs.append(np.load(os.path.join(p_dir, f)))

        if len(embs) >= 3:
            emb_arr = np.vstack(embs)
            stats[person] = {
                "center": np.mean(emb_arr, axis=0),
                "std": np.std(emb_arr, axis=0).mean()
            }

    return stats

# ===============================================================
# 重新訓練 SVM / KNN（三分類器）
# ===============================================================
def retrain_models(cache_root=CACHE_ROOT):

    X = []
    y = []

    for person in sorted(os.listdir(cache_root)):
        p_dir = os.path.join(cache_root, person)
        if not os.path.isdir(p_dir):
            continue

        for f in os.listdir(p_dir):
            if f.endswith(".npy"):
                emb = np.load(os.path.join(p_dir, f))
                X.append(emb)
                y.append(person)

    X = np.array(X)
    y = np.array(y)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)
    pickle.dump(svm, open(os.path.join(cache_root, "svm.pkl"), "wb"))

    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    knn.fit(X, y)
    pickle.dump(knn, open(os.path.join(cache_root, "knn.pkl"), "wb"))

    return svm, knn


# ===============================================================
# Ensemble（三分類器投票）
# ===============================================================
def ensemble_predict(emb, db, svm, knn, cos_threshold=0.38):

    best_person, best_score = "Unknown", -1
    for person, center in db.items():
        score = float(np.dot(emb, center))
        if score > best_score:
            best_score = score
            best_person = person

    cosine_pred = best_person if best_score >= cos_threshold else "Unknown"

    # SVM
    svm_pred = svm.predict([emb])[0]
    svm_conf = max(svm.predict_proba([emb])[0])

    # KNN
    knn_pred = knn.predict([emb])[0]

    votes = [cosine_pred, svm_pred, knn_pred]
    final = max(votes, key=votes.count)

    return final, {
        "cosine_pred": cosine_pred,
        "cosine_conf": best_score,
        "svm_pred": svm_pred,
        "svm_conf": svm_conf,
        "knn_pred": knn_pred
    }


# ===============================================================
# 【新增】Cloud Run 專用 — 單張臉辨識
# ===============================================================
def predict_single(img):
    """
    Cloud Run API 用
    img: OpenCV BGR 圖片
    return: dict
    """
    global database, svm_model, knn_model

    faces = app.get(img)
    if len(faces) == 0:
        return {"error": "No face detected"}

    f = faces[0]
    emb = f.normed_embedding

    final_pred, details = ensemble_predict(
        emb,
        database,
        svm_model,
        knn_model
    )

    return {
        "final_pred": final_pred,
        "details": {
            "cosine_pred": details["cosine_pred"],
            "cosine_conf": float(details["cosine_conf"]),
            "svm_pred": details["svm_pred"],
            "svm_conf": float(details["svm_conf"]),
            "knn_pred": details["knn_pred"]
        }
    }


# ===============================================================
# 【新增】Cloud Run 專用 — 團體照 API（選用）
# ===============================================================
def predict_group(img):
    """
    多張臉（團體照）辨識 API
    """
    global database, svm_model, knn_model

    faces = app.get(img)
    results = []

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        crop = img[y1:y2, x1:x2]
        emb = f.normed_embedding

        final_pred, details = ensemble_predict(
            emb,
            database,
            svm_model,
            knn_model
        )

        results.append({
            "bbox": [x1, y1, x2, y2],
            "final_pred": final_pred,
            "details": {
                "cosine_pred": details["cosine_pred"],
                "cosine_conf": float(details["cosine_conf"]),
                "svm_pred": details["svm_pred"],
                "svm_conf": float(details["svm_conf"]),
                "knn_pred": details["knn_pred"]
            }
        })

    return results


# ===============================================================
# 初始化（Cloud Run 不會跑 retrain）
# ===============================================================
# 載入模型（你 GitHub src/models 放的）
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

svm_model = pickle.load(open(os.path.join(MODELS_DIR, "svm.pkl"), "rb"))
knn_model = pickle.load(open(os.path.join(MODELS_DIR, "knn.pkl"), "rb"))
database = pickle.load(open(os.path.join(MODELS_DIR, "centers.pkl"), "rb"))

# ===============================================================
# 完成
# ===============================================================
print("✅ auto_train_group.py 已完成初始化（Cloud Run 模式）")
