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
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

# ================================
# 1. 載入舊資料（X, y_raw, label_map）
# ================================
def load_old_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")
    map_path = os.path.join(MODEL_DIR, "label_map.json")

    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(map_path):
        print("📂 載入舊資料 X.npy / y.npy / label_map.json")

        X = np.load(X_path)
        y_index = np.load(y_path)
        label_map = json.load(open(map_path, "r", encoding="utf-8"))

        inv_map = {int(v): k for k, v in label_map.items()}
        y_raw = np.array([inv_map[idx] for idx in y_index])

        return X, y_raw, label_map

    print("⚠️ 第一次訓練，未找到舊模型")
    return np.array([]), np.array([]), {}

# ================================
# 2. 偵測新增 / 刪除人員
# ================================
def detect_person_change(label_map_old):
    old_names = set(label_map_old.keys())
    current_names = set(os.listdir(RAW_DIR))

    deleted = old_names - current_names
    added = current_names - old_names

    changed = False

    if deleted:
        print(f"⚠️ 偵測到人物被刪除：{deleted}")
        changed = True
    if added:
        print(f"🆕 偵測到新增人物：{added}")
        changed = True

    return changed

# ================================
# 3. 偵測照片數量變動
# ================================
def detect_image_count_changed():
    record_path = os.path.join(MODEL_DIR, "image_count.json")

    if not os.path.exists(record_path):
        return True

    old_record = json.load(open(record_path, "r", encoding="utf-8"))
    new_record = {}
    changed = False

    for person in os.listdir(RAW_DIR):
        p_dir = os.path.join(RAW_DIR, person)
        if not os.path.isdir(p_dir):
            continue

        count = len(os.listdir(p_dir))
        new_record[person] = count

        if person not in old_record or old_record[person] != count:
            print(f"⚠️ {person} 的照片數量改變，需要重新訓練")
            changed = True

    json.dump(new_record, open(record_path, "w", encoding="utf-8"), indent=2)
    return changed

# ================================
# 4. 提取 embeddings
# ================================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def extract_embeddings(person_list):
    X, y = [], []
    for person in person_list:
        p_dir = os.path.join(RAW_DIR, person)
        imgs = os.listdir(p_dir)

        print(f"\n📸 {person} — {len(imgs)} 張圖片")

        for img in tqdm(imgs):
            path = os.path.join(p_dir, img)
            img = imread_safe(path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            X.append(faces[0].normed_embedding)
            y.append(person)

    return np.array(X), np.array(y)

# ================================
# 5. 三分類器
# ================================
def train_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

def train_svm(X, y):
    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)
    return model

def calc_centers(X, y):
    centers = {}
    for person in np.unique(y):
        centers[person] = X[y == person].mean(axis=0)
    return centers

# ================================
# 6. 自動 threshold（距離版）
# ================================
def auto_threshold_distance(X, y):
    same_dists = []
    diff_dists = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.linalg.norm(X[i] - X[j])
            if y[i] == y[j]:
                same_dists.append(d)
            else:
                diff_dists.append(d)

    same_dists = np.array(same_dists)
    diff_dists = np.array(diff_dists)

    print(f"✔ SAME avg：{same_dists.mean():.4f}")
    print(f"❌ DIFF avg：{diff_dists.mean():.4f}")

    # Youden’s J
    candidates = np.linspace(0, 2, 2000)
    best_j = -1
    best_t = 0

    for t in candidates:
        tp = np.sum(same_dists <= t)
        fn = np.sum(same_dists > t)
        tn = np.sum(diff_dists > t)
        fp = np.sum(diff_dists <= t)

        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        J = sens + spec - 1

        if J > best_j:
            best_j = J
            best_t = t

    thresholds = {
        "conservative": float(same_dists.max() + 0.02),
        "balanced": float(best_t),
        "loose": float(diff_dists.min() - 0.02)
    }

    json.dump(thresholds, open(os.path.join(MODEL_DIR, "threshold.json"), "w"), indent=2)
    return thresholds

# ================================
# 7. 儲存
# ================================
def save_all(X_raw, y_index, knn, svm, centers, label_map, thresholds):
    np.save(f"{MODEL_DIR}/X.npy", X_raw)
    np.save(f"{MODEL_DIR}/y.npy", y_index)

    pickle.dump(knn, open(f"{MODEL_DIR}/knn.pkl", "wb"))
    pickle.dump(svm, open(f"{MODEL_DIR}/svm.pkl", "wb"))
    pickle.dump(centers, open(f"{MODEL_DIR}/centers.pkl", "wb"))

    json.dump(label_map, open(f"{MODEL_DIR}/label_map.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n💾 模型與資料保存完成！")

# ================================
# 8. Main
# ================================
if __name__ == "__main__":
    X_old, y_old, label_map_old = load_old_data()

    must_retrain = False

    # 人名變動
    if detect_person_change(label_map_old):
        must_retrain = True

    # 照片數變動
    if detect_image_count_changed():
        must_retrain = True

    if not must_retrain and len(X_old) > 0:
        print("\n✔️ 沒有任何變化，不需要重新訓練")
        exit()

    # --- 開始訓練 ---
    persons = sorted(os.listdir(RAW_DIR))
    X_new, y_new = extract_embeddings(persons)

    # 新 label_map
    unique_names = sorted(set(y_new.tolist()))
    label_map = {name: idx for idx, name in enumerate(unique_names)}
    y_index = np.array([label_map[name] for name in y_new])

    knn = train_knn(X_new, y_new)
    svm = train_svm(X_new, y_new)
    centers = calc_centers(X_new, y_new)

    thresholds = auto_threshold_distance(X_new, y_new)

    save_all(X_new, y_index, knn, svm, centers, label_map, thresholds)

    print("\n🎉 三分類器訓練完成！")
