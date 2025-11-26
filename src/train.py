import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ============================================================
# 🔧 1. 可切換模型（buffalo_l / buffalo_sc / antelope_v2）
# ============================================================
MODEL_NAME = "antelope_v2"     # ← 你可改 buffalo_l / buffalo_sc

# ============================================================
# 🔧 2. 資料與輸出位置
# ============================================================
RAW_DIR = "/content/drive/MyDrive/face_DataSet/face_raw" 
MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


# ============================================================
# 🔥 Step 1 — 提取所有 Embeddings
# ============================================================
def extract_embeddings():
    print(f"\n🚀 使用模型：{MODEL_NAME}")

    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=0)

    X = []
    y = []

    persons = sorted(os.listdir(RAW_DIR))
    print("\n📂 偵測到人物資料夾：", persons)

    for person in persons:
        p_dir = os.path.join(RAW_DIR, person)
        if not os.path.isdir(p_dir):
            continue

        images = os.listdir(p_dir)
        print(f"\n📸 {person}: {len(images)} 張")

        for img_name in tqdm(images):
            img_path = os.path.join(p_dir, img_name)
            img = imread_safe(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            emb = faces[0].normed_embedding
            X.append(emb)
            y.append(person)

    X = np.array(X)
    y = np.array(y)

    print("\n✔ 產生 embedding：", X.shape)
    return X, y


# ============================================================
# 🔥 Step 2 — 儲存工具
# ============================================================
def save_pickle(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"💾 saved: {path}")


# ============================================================
# 🔥 Step 3 — 訓練三分類器
# ============================================================
def train_all():
    # -----------------------------------------
    # Step 1：Extract Embeddings
    # -----------------------------------------
    X, y = extract_embeddings()

    # -----------------------------------------
    # Step 2：Train KNN
    # -----------------------------------------
    print("\n🚀 Training KNN ...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    # -----------------------------------------
    # Step 3：Train SVM
    # -----------------------------------------
    print("\n🚀 Training SVM ...")
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)

    # -----------------------------------------
    # Step 4：Compute Cosine Centers
    # -----------------------------------------
    print("\n🚀 Computing Class Centers ...")
    centers = {}
    for person in np.unique(y):
        centers[person] = X[y == person].mean(axis=0)

    # -----------------------------------------
    # Step 5：Save all models
    # -----------------------------------------
    print("\n💾 Saving all models...")

    save_pickle(knn, "knn.pkl")
    save_pickle(svm, "svm.pkl")
    save_pickle(centers, "centers.pkl")

    # 附加儲存訓練資料（用於 t-SNE）
    np.save(os.path.join(MODEL_DIR, "X.npy"), X)
    np.save(os.path.join(MODEL_DIR, "y.npy"), y)

    print("\n🎉 完成！模型全部訓練成功！")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_all()
