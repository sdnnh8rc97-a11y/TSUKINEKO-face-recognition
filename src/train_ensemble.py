%%writefile /content/face_system/src/train_ensemble.py
import os
import json
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from insightface.app import FaceAnalysis


# ================================
# 設定
# ================================
DATA_DIR = "/content/drive/MyDrive/face_DataSet"
RAW_DIR = f"{DATA_DIR}/face_raw"
MODEL_DIR = "/content/face_system/models"

os.makedirs(MODEL_DIR, exist_ok=True)


def imread_safe(path):
    """支援中文/空格路徑"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


# ================================
# 偵測新增人物
# ================================
def load_old_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("📂 載入舊資料 X.npy / y.npy")
        return np.load(X_path), np.load(y_path)
    print("⚠️ 無舊資料，第一次訓練")
    return np.array([]), np.array([])


def detect_new_persons(y_old):
    """檢查 RAW_DIR 裡哪些資料夾沒訓練過"""
    persons = sorted(os.listdir(RAW_DIR))
    old_people = set(y_old.tolist()) if len(y_old) > 0 else set()

    new_list = [p for p in persons if p not in old_people]
    print(f"\n🆕 新增人物：{new_list}")
    return new_list


# ================================
# 建立 embeddings
# ================================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)


def extract_embeddings(person_list):
    X, y = [], []

    for person in person_list:
        p_dir = os.path.join(RAW_DIR, person)
        images = os.listdir(p_dir)

        print(f"\n📸 {person} — {len(images)} 張圖片")

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


# ================================
# 3 分類器訓練
# ================================
def train_knn(X, y):
    print("\n🚀 訓練 KNN ...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn


def train_svm(X, y):
    print("\n🚀 訓練 SVM ...")
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)
    return svm


def calc_centers(X, y):
    print("\n🚀 計算 Centers ...")
    centers = {}
    labels = np.unique(y)
    for person in labels:
        centers[person] = X[y == person].mean(axis=0)
    return centers


# ================================
# 門檻自動微調（Unknown 最重要的部分）
# ================================
def compute_cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def auto_threshold(X, y, centers):
    print("\n🧠 自動微調 Unknown 門檻 ...")

    pos = []
    neg = []

    for emb, label in zip(X, y):
        # 正例 similarity
        pos.append(compute_cosine(emb, centers[label]))

        # 負例 similarity
        for other, vec in centers.items():
            if other != label:
                neg.append(compute_cosine(emb, vec))

    pos = np.array(pos)
    neg = np.array(neg)

    # 門檻建議作法：負例的 μ + 1.5σ
    thr = neg.mean() + 1.5 * neg.std()

    thr = float(max(min(thr, 0.60), 0.30))  # 安全限制區間
    print(f"📌 建議門檻：{thr:.4f}")

    return thr


# ================================
# 儲存
# ================================
def save_all(X, y, knn, svm, centers, label_map, threshold):
    np.save(f"{MODEL_DIR}/X.npy", X)
    np.save(f"{MODEL_DIR}/y.npy", y)

    with open(f"{MODEL_DIR}/knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open(f"{MODEL_DIR}/svm.pkl", "wb") as f:
        pickle.dump(svm, f)

    with open(f"{MODEL_DIR}/centers.pkl", "wb") as f:
        pickle.dump(centers, f)

    with open(f"{MODEL_DIR}/label_map.json", "w") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    with open(f"{MODEL_DIR}/threshold.json", "w") as f:
        json.dump({"cosine_threshold": threshold}, f, indent=2)

    print("\n💾 所有模型/資料已保存完畢！")


# ================================
# Main
# ================================
if __name__ == "__main__":
    X_old, y_old = load_old_data()
    new_list = detect_new_persons(y_old)

    if len(new_list) == 0 and len(X_old) > 0:
        print("\n✔ 沒有新增人員，不需要重新訓練")
        exit()

    X_new, y_new = extract_embeddings(new_list)

    # 合併資料
    X = np.concatenate([X_old, X_new]) if len(X_old) > 0 else X_new
    y = np.concatenate([y_old, y_new]) if len(y_old) > 0 else y_new

    # label_map
    label_map = {label: i for i, label in enumerate(sorted(np.unique(y)))}

    # 訓練三分類器
    knn = train_knn(X, y)
    svm = train_svm(X, y)
    centers = calc_centers(X, y)

    # 動態 Unknown 門檻
    thr = auto_threshold(X, y, centers)

    # 儲存全部模型
    save_all(X, y, knn, svm, centers, label_map, thr)

    print("\n🎉 三分類器訓練完成！")
