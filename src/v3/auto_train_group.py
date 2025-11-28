# ===============================================================
#  三合一臉辨系統（高速版 + Anti-Drift Auto-Train + 團體照分類）
#  作者：まさき專用（V3.1 安全版）
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
# 設定路徑（可自行調整）
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

    print("📌 偵測到人員資料夾：", persons)

    for person in persons:
        raw_dir = os.path.join(RAW_ROOT, person)
        cache_dir = os.path.join(CACHE_ROOT, person)
        os.makedirs(cache_dir, exist_ok=True)

        photos = [
            f for f in os.listdir(raw_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        print(f"\n==============================")
        print(f"👤 {person} — {len(photos)} 張照片")
        print("==============================")

        for img_name in tqdm(photos, desc=f"建立 cache：{person}", ncols=80):
            raw_path = os.path.join(raw_dir, img_name)
            cache_path = os.path.join(cache_dir, img_name + ".npy")

            # cache 已存在 → 跳過（秒跑）
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

    print("\n🎉 STEP1 完成：cache 建立完畢！")


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
            print(f"✔ 資料庫：{person}（{len(embs)} 筆 embedding）")

    return db


# ===============================================================
#  STEP 2.1 — Cluster Stats（中心＋標準差）
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
# 自動更新 cache（raw+embedding）
# ===============================================================
def update_cache_for(person, emb, filename):
    cache_dir = os.path.join(CACHE_ROOT, person)
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, filename + ".npy")
    np.save(cache_path, emb)

    print(f"🔄 Auto-Cache：已寫入 → {cache_path}")


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

    print(f"📌 retrain 樣本數：{len(X)}")

    # Train SVM
    print("🔧 訓練 SVM ...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)
    pickle.dump(svm, open(os.path.join(cache_root, "svm.pkl"), "wb"))

    # Train KNN
    print("🔧 訓練 KNN ...")
    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    knn.fit(X, y)
    pickle.dump(knn, open(os.path.join(cache_root, "knn.pkl"), "wb"))

    print("🎉 Retrain 完成！")
    return svm, knn


# 先載入一次
svm_model, knn_model = retrain_models()


# ===============================================================
# Ensemble（三分類器投票）
# ===============================================================
def ensemble_predict(emb, db, svm, knn, cos_threshold=0.38):

    # Cosine
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

    # Voting
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
# V3.1 Auto-Train 判斷（企業級 Anti-Drift）
# ===============================================================
def allow_auto_train(final_pred, details, emb, db_stats):

    cosine_ok = details["cosine_conf"] >= 0.78
    svm_ok = details["svm_conf"] >= 0.85

    consistent = (
        details["cosine_pred"] == final_pred and
        details["svm_pred"] == final_pred and
        details["knn_pred"] == final_pred
    )

    if not (cosine_ok and svm_ok and consistent):
        return False

    # Cluster Distance Check
    center = db_stats[final_pred]["center"]
    std = db_stats[final_pred]["std"]

    dist = np.linalg.norm(emb - center)
    max_allowed = std * 1.2

    return dist <= max_allowed


# ===============================================================
# STEP 3 — 團體照分類（含 Auto-Train V3.1）
# ===============================================================
def classify_group_photo():
    global svm_model, knn_model

    os.makedirs(CLASSIFY_SAVE, exist_ok=True)

    img = cv2.imread(GROUP_PHOTO)
    faces = app.get(img)

    faces = sorted(faces, key=lambda f: f.bbox[0])  # 左→右排序
    db_stats = build_db_stats()

    print(f"\n📸 偵測到 {len(faces)} 張臉（已排序）\n")

    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)
        crop = img[y1:y2, x1:x2]
        emb = f.normed_embedding

        final_pred, details = ensemble_predict(emb, database, svm_model, knn_model)

        save_dir = os.path.join(CLASSIFY_SAVE, final_pred)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"group_{i+1}.jpg")
        cv2.imwrite(out_path, crop)

        print(f"臉 {i+1}: 最終分類 → {final_pred}")
        print(details)

        # --- Auto-Train V3.1 ---
        if final_pred != "Unknown" and final_pred in db_stats:
            if allow_auto_train(final_pred, details, emb, db_stats):

                auto_raw_dir = os.path.join(RAW_ROOT, final_pred)
                os.makedirs(auto_raw_dir, exist_ok=True)

                add_path = os.path.join(auto_raw_dir, f"auto_{i+1}.jpg")
                cv2.imwrite(add_path, crop)

                print(f"✅ Auto-Train：新增 raw → {add_path}")

                update_cache_for(final_pred, emb, f"auto_{i+1}")

            else:
                print("⚠️ Auto-Train 跳過（信心或 cluster 距離不足）")
        else:
            print("⚠️ 未加入 Auto-Train（Unknown 或 stats 不足）")

    # Retrain 一次
    print("\n🔄 Auto retrain（三分類器）...")
    svm_model, knn_model = retrain_models()
    print("🎉 Auto retrain 完成！模型已更新")

    print("\n🎉 STEP3 完成：團體照分類完畢！")


# ===============================================================
#  一鍵執行全部步驟
# ===============================================================
print("\n🚀 STEP1：開始建立 embedding cache ...")
build_cache()

print("\n🚀 STEP2：建立資料庫 ...")
database = load_database()

print("\n🚀 STEP3：開始處理團體照 ...")
classify_group_photo()

print("\n🎉 全流程完成！")

