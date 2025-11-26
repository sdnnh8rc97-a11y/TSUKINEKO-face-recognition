import os
import json
import joblib
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# 引入你的 embedder
from face_embedder import FaceEmbedder

CLEAN_DIR = "/content/drive/MyDrive/face_DataSet/face_clean"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

embedder = FaceEmbedder()

# -------------------------
# Step 1 — 建立 Embeddings
# -------------------------
def load_embeddings():
    X = []
    y = []
    centers = {}

    persons = sorted(os.listdir(CLEAN_DIR))

    for person in persons:
        person_dir = os.path.join(CLEAN_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        imgs = glob(os.path.join(person_dir, "*.jpg")) + glob(os.path.join(person_dir, "*.png"))
        person_embs = []

        print(f"Processing: {person} ({len(imgs)} images)")

        for img_path in imgs:
            emb = embedder.get_embedding(img_path)
            if emb is None:
                continue

            X.append(emb)
            y.append(person)
            person_embs.append(emb)

        # 計算中心向量
        if len(person_embs) > 0:
            centers[person] = np.mean(person_embs, axis=0).tolist()

    return np.array(X), np.array(y), centers


print("🔵 STEP 1 — 建立 embeddings")
X, y, centers = load_embeddings()

np.save(os.path.join(MODEL_DIR, "X.npy"), X)
np.save(os.path.join(MODEL_DIR, "y.npy"), y)
json.dump(centers, open(os.path.join(MODEL_DIR, "centers.json"), "w"), indent=4)

print("✔ embeddings 完成")
print()

# -------------------------
# Step 2 — Label Encoding
# -------------------------
print("🔵 STEP 2 — 編碼 label")

le = LabelEncoder()
y_num = le.fit_transform(y)

label_map = {int(i): name for i, name in enumerate(le.classes_)}
json.dump(label_map, open(os.path.join(MODEL_DIR, "label_map.json"), "w"), indent=4)

print("✔ label map 完成")
print()

# -------------------------
# Step 3 — 訓練 KNN
# -------------------------
print("🔵 STEP 3 — 訓練 KNN")

knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
knn.fit(X, y_num)

joblib.dump(knn, os.path.join(MODEL_DIR, "knn.pkl"))
print("✔ KNN 訓練完成")
print()

# -------------------------
# Step 4 — 訓練 SVM
# -------------------------
print("🔵 STEP 4 — 訓練 SVM")

svm = SVC(kernel="linear", probability=True)
svm.fit(X, y_num)

joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
print("✔ SVM 訓練完成")
print()

print("🎉 全部模型訓練完成！")
