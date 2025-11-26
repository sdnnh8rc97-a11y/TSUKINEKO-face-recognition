import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ============================================================
# 🔧 模型名稱（建議 buffalo_l）
# ============================================================
MODEL_NAME = "buffalo_l"

# ============================================================
# 🔧 1. 原完整訓練集（舊保全）
# 🔧 2. 新增照片資料夾（只放新保全）
# ============================================================
RAW_DIR = "/content/drive/MyDrive/face_DataSet/face_raw"          # 舊的完整資料
NEW_DIR = "/content/drive/MyDrive/face_DataSet/face_new"          # 只放新增保全的照片

MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


# ============================================================
# 🔥 Step 1：讀取舊 embeddings
# ============================================================
def load_old_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("❌ 找不到舊資料，請先做完整訓練 train.py")
        exit()

    X_old = np.load(X_path)
    y_old = np.load(y_path)

    print("📂 載入舊資料：", X_old.shape)
    return X_old, y_old


# ============================================================
# 🔥 Step 2：讀取增量資料（只跑 new 資料夾）
# ============================================================
def load_new_embeddings():
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=0)

    X_new = []
    y_new = []

    persons = sorted(os.listdir(NEW_DIR))
    print("\n🆕 偵測到新增人物資料夾：", persons)

    for person in persons:
        p_dir = os.path.join(NEW_DIR, person)
        if not os.path.isdir(p_dir):
            continue

        images = os.listdir(p_dir)
        print(f"\n📸 新增 {person}: {len(images)} 張")

        for img_name in tqdm(images):
            img_path = os.path.join(p_dir, img_name)
            img = imread_safe(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if faces:
                X_new.append(faces[0].normed_embedding)
                y_new.append(person)

    return np.array(X_new), np.array(y_new)


# ============================================================
# 🔥 Step 3：重訓 KNN / SVM / Centers
# ============================================================
def save_pickle(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("💾 saved:", path)


def train_incremental():
    # 讀取舊資料
    X_old, y_old = load_old_data()

    # 讀取新資料
    X_new, y_new = load_new_embeddings()

    # 合併
    X = np.concatenate([X_old, X_new], axis=0)
    y = np.concatenate([y_old, y_new], axis=0)

    print("\n📦 合併後資料量：", X.shape)

    # 訓練 KNN
    print("\n🚀 Training KNN ...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    # 訓練 SVM
    print("\n🚀 Training SVM ...")
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)

    # 計算 Cosine Centers
    print("\n🚀 Updating Centers ...")
    centers = {}
    for person in np.unique(y):
        centers[person] = X[y == person].mean(axis=0)

    # 儲存
    print("\n💾 Saving updated models...")
    save_pickle(knn, "knn.pkl")
    save_pickle(svm, "svm.pkl")
    save_pickle(centers, "centers.pkl")

    # 記得更新 X / y
    np.save(os.path.join(MODEL_DIR, "X.npy"), X)
    np.save(os.path.join(MODEL_DIR, "y.npy"), y)

    print("\n🎉 增量訓練完成！")


if __name__ == "__main__":
    train_incremental()
