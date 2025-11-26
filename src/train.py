import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

RAW_DIR = "/content/drive/MyDrive/face_DataSet/face_raw"   # 你之後可改參數
MODEL_DIR = "src/models"

os.makedirs(MODEL_DIR, exist_ok=True)

def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

def extract_embeddings():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)

    X = []
    y = []

    persons = sorted(os.listdir(RAW_DIR))

    for person in persons:
        p_dir = os.path.join(RAW_DIR, person)
        if not os.path.isdir(p_dir):
            continue

        images = os.listdir(p_dir)
        print(f"📸 {person}: {len(images)} 張")

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

    return np.array(X), np.array(y)

def save_model(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    np.save(path, obj)
    print(f"💾 saved: {path}")

def train_all():
    print("\n🚀 Step1 — Extract embeddings")
    X, y = extract_embeddings()

    print("\n🚀 Step2 — Train KNN")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    print("\n🚀 Step3 — Train SVM")
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)

    print("\n🚀 Step4 — Compute class centers")
    centers = {}
    for person in np.unique(y):
        centers[person] = X[y == person].mean(axis=0)

    print("\n💾 Saving all models...")
    save_model(knn, "knn.npy")
    save_model(svm, "svm.npy")
    save_model(centers, "centers.npy")

    print("\n✅ 全部完成！")

if __name__ == "__main__":
    train_all()
