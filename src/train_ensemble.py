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
# 載入舊資料
# ================================
def load_old_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("📂 載入舊資料 X.npy / y.npy")
        return np.load(X_path), np.load(y_path)
    print("⚠️ 無舊資料，第一次訓練")
    return np.array([]), np.array([])


def load_old_data():
    X_path = os.path.join(MODEL_DIR, "X.npy")
    y_path = os.path.join(MODEL_DIR, "y.npy")
    map_path = os.path.join(MODEL_DIR, "label_map.json")

    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(map_path):
        print("📂 載入舊資料 X.npy / y.npy / label_map.json")

        X = np.load(X_path)
        y_index = np.load(y_path)
        label_map = json.load(open(map_path, "r", encoding="utf-8"))

        # 反查 index → 中文名字
        inv_map = {v: k for k, v in label_map.items()}
        y_raw = np.array([inv_map[str(idx)] for idx in y_index])

        return X, y_raw, label_map

    print("⚠️ 第一次訓練，未找到舊模型")
    return np.array([]), np.array([]), {}


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
# 三分類器
# ================================
def train_knn(X, y):
    print("\n🚀 訓練 KNN ...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

def train_svm(X, y):
    print("\n🚀 訓練 SVM ...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)
    return model

def calc_centers(X, y):
    print("\n🚀 計算 centers ...")
    centers = {}
    labels = np.unique(y)
    for person in labels:
        centers[person] = X[y == person].mean(axis=0)
    return centers


# ================================
# ⭐⭐⭐ 自動 threshold（距離版）⭐⭐⭐
# ================================
def auto_threshold_distance(X, y):
    print("\n📊 正在載入 embedding X, y ...")
    same_dists = []
    diff_dists = []

    print("📏 計算 SAME / DIFF 距離中...\n")

    # 全部 pairwise distance
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.linalg.norm(X[i] - X[j])  # L2 distance

            if y[i] == y[j]:
                same_dists.append(d)
            else:
                diff_dists.append(d)

    same_dists = np.array(same_dists)
    diff_dists = np.array(diff_dists)

    print(f"✔ SAME（同一人）距離")
    print(f"   平均：{same_dists.mean():.4f}")
    print(f"   最小：{same_dists.min():.4f}")
    print(f"   最大：{same_dists.max():.4f}\n")

    print(f"❌ DIFF（不同人）距離")
    print(f"   平均：{diff_dists.mean():.4f}")
    print(f"   最小：{diff_dists.min():.4f}")
    print(f"   最大：{diff_dists.max():.4f}\n")

    # ======== 偵測嚴重錯誤（不同人距離 = 0）========
    print("🕵️‍♂️ 檢查是否有 DIFF 距離 = 0 ...")
    zero_dist_indices = np.where(diff_dists == 0)[0]
    if len(zero_dist_indices) > 0:
        print("❗ 注意：有不同人的 embedding 完全相同！")
        print("   ➤ 代表照片資料錯放 or embedding 錯混")
    else:
        print("✔ 未發現距離=0 的異常 embedding")

    # ======== Youden’s J 最佳 threshold ========
    print("\n🔍 正在使用 Youden’s J 找最佳 threshold...\n")

    candidates = np.linspace(0.0, 2.0, 2000)
    best_j = -1
    best_t = 0

    for t in candidates:
        tp = np.sum(same_dists <= t)
        fn = np.sum(same_dists > t)
        tn = np.sum(diff_dists > t)
        fp = np.sum(diff_dists <= t)

        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        J = sensitivity + specificity - 1

        if J > best_j:
            best_j = J
            best_t = t

    # 三種策略
    t_conservative = same_dists.max() + 0.02
    t_balanced = best_t
    t_loose = diff_dists.min() - 0.02

    print("🎯 自動 threshold 計算結果：\n")
    print(f"🔒 保守（不錯認）：{t_conservative:.4f}")
    print(f"⚖️ 平衡（最佳 J）：{t_balanced:.4f}")
    print(f"🎈 寬鬆（不漏認）：{t_loose:.4f}")

    # 寫檔
    thresholds = {
        "conservative": float(t_conservative),
        "balanced": float(t_balanced),
        "loose": float(t_loose)
    }

    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump(thresholds, f, indent=4)

    print("\n💾 已寫入 threshold.json")
    return thresholds


# ================================
# 儲存
# ================================
def save_all(X, y, knn, svm, centers, label_map, thresholds):
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

    print("\n💾 所有模型/資料已保存完畢！")


# ================================
# Main
# ================================
if __name__ == "__main__":
    X_old, y_old, label_map_old = load_old_data()

    # 偵測新增人物
    existing_names = set(y_old.tolist())
    new_list = detect_new_persons(existing_names)

    if len(new_list) == 0 and len(X_old) > 0:
        print("\n✔ 沒有新增人員，不需要重新訓練")
        exit()

    # 提取 embedding
    X_new, y_new = extract_embeddings(new_list)

    # 合併
    X_raw = np.concatenate([X_old, X_new]) if len(X_old) > 0 else X_new
    y_raw = np.concatenate([y_old, y_new]) if len(y_old) > 0 else y_new

    # 重建 label_map（中文 → index）
    unique_names = sorted(set(y_raw.tolist()))
    label_map = {name: idx for idx, name in enumerate(unique_names)}
    y_index = np.array([label_map[name] for name in y_raw])

    # 重新訓練分類器
    knn = train_knn(X_raw, y_raw)
    svm = train_svm(X_raw, y_raw)
    centers = calc_centers(X_raw, y_raw)

    # 自動 threshold
    thresholds = auto_threshold_distance(X_raw, y_raw)

    # 儲存
    save_all(X_raw, y_index, knn, svm, centers, label_map, thresholds)

    print("\n🎉 三分類器訓練完成！")
