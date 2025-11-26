import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


MODEL_NAME = "buffalo_l"
RAW_DIR = "/content/drive/MyDrive/face_DataSet/face_raw"
MODEL_DIR = "src/models"

os.makedirs(MODEL_DIR, exist_ok=True)


def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


# ============================================================
# 讀取舊資料（如不存在則視為第一次訓練）
# ============================================================
def load_old_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("📂 載入舊資料 X, y")
        X_old = np.load(X_path)
        y_old = np.load(y_path)
        return X_old, y_old
    else:
        print("⚠️ 找不到舊資料，視為第一次訓練")
        return np.array([]), np.array([])


# ============================================================
# 偵測哪些是「新增照片」
# ============================================================
def detect_new_images(y_old):
    persons = sorted(os.listdir(RAW_DIR))
    old_people = set(y_old.tolist()) if len(y_old) > 0 else set()

    new_list = []
    for person in persons:
        if person not in old_people:
            new_list.append(person)

    print("\n🆕 新增人員：", new_list)
    return new_list


# ============================================================
# 為指定人員建立 embedding（只跑增量）
# ============================================================
def extract_embeddings_for(person_list):
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=0)

    X_new = []
    y_new = []

    for person in person_list:
        p_dir = os.path.join(RAW_DIR, person)
        images = os.listdir(p_dir)

        print(f"\n📸 增量人物 {person}: {len(images)} 張")

        for img_name in tqdm(images):
            img_path = os.path.join(p_dir, img_name)
            img = imread_safe(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            emb = faces[0].normed_embedding
            X_new.append(emb)
            y_new.append(person)

    return np.array(X_new), np.array(y_new)


# ============================================================
# 重訓三分類器（使用新舊混合資料）
# ============================================================
def retrain_models(X, y):
    print("\n🚀 Retrain KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    print("\n🚀 Retrain SVM...")
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)

    print("\n🚀 Update Centers...")
    centers = {}
    for person in np.unique(y):
        centers[person] = X[y == person].mean(axis=0)

    return knn, svm, centers


# ============================================================
# 儲存
# ============================================================
def save_all(X, y, knn, svm, centers):
    np.save(os.path.join(MODEL_DIR, "X.npy"), X)
    np.save(os.path.join(MODEL_DIR, "y.npy"), y)

    with open(os.path.join(MODEL_DIR, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)

    with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
        pickle.dump(svm, f)

    with open(os.path.join(MODEL_DIR, "centers.pkl"), "wb") as f:
        pickle.dump(centers, f)

    print("\n💾 模型已更新（增量完成）")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    X_old, y_old = load_old_data()
    new_persons = detect_new_images(y_old)

    if len(new_persons) == 0:
        print("\n✔ 沒有新增人物，不需要增量訓練")
        exit()

    X_new, y_new = extract_embeddings_for(new_persons)

    X = np.concatenate([X_old, X_new]) if len(X_old) > 0 else X_new
    y = np.concatenate([y_old, y_new]) if len(y_old) > 0 else y_new

    knn, svm, centers = retrain_models(X, y)
    save_all(X, y, knn, svm, centers)

    print("\n🎉 增量訓練完成！")
