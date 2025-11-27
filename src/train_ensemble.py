import os
import json
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ======================================================
# 基本設定
# ======================================================
DATA_DIR = "/content/drive/MyDrive/face_DataSet"
RAW_DIR = f"{DATA_DIR}/face_raw"
MODEL_DIR = f"{DATA_DIR}/models"

os.makedirs(MODEL_DIR, exist_ok=True)

def imread_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

# ======================================================
# 載入舊資料
# ======================================================
def load_old_embeddings():
    X_path = f"{MODEL_DIR}/X.npy"
    y_path = f"{MODEL_DIR}/y.npy"
    map_path = f"{MODEL_DIR}/label_map.json"

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(map_path)):
        print("📂 第一次訓練：沒有舊資料")
        return np.array([]), np.array([]), {}

    print("📂 載入舊的 X.npy / y.npy / label_map.json")
    X = np.load(X_path)
    y = np.load(y_path)     # y 儲存 index（以 label_map 對應）
    label_map = json.load(open(map_path, "r", encoding="utf-8"))

    return X, y, label_map

# ======================================================
# 找出所有變動的人
# ======================================================
def detect_changes(label_map_old):
    current_people = sorted(os.listdir(RAW_DIR))
    old_people = sorted(list(label_map_old.keys()))

    added = list(set(current_people) - set(old_people))
    deleted = list(set(old_people) - set(current_people))
    same_people = list(set(current_people) & set(old_people))

    # 比較各自照片數量
    changed = []
    for person in same_people:
        raw_count = len(os.listdir(f"{RAW_DIR}/{person}"))
        # 老 label_map 裡是 index，找不到照片數 → 視為變動
        # 我們從 image_count.json 記錄數量
        pass

    # 使用 image_count.json 追蹤照片數變化
    count_file = f"{MODEL_DIR}/image_count.json"
    old_count = {}
    if os.path.exists(count_file):
        old_count = json.load(open(count_file, "r"))
    else:
        old_count = {}

    new_count = {}
    for person in current_people:
        new_count[person] = len(os.listdir(f"{RAW_DIR}/{person}"))
        if person not in old_count or old_count[person] != new_count[person]:
            changed.append(person)

    json.dump(new_count, open(count_file, "w"), indent=2, ensure_ascii=False)

    print(f"🆕 新增人員：{added}")
    print(f"❌ 刪除人員：{deleted}")
    print(f"♻️ 照片數變動：{changed}")

    return added, deleted, changed

# ======================================================
# 抽取 embeddings（針對某一個人）
# ======================================================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def extract_person_embeddings(person):
    p_dir = f"{RAW_DIR}/{person}"
    imgs = os.listdir(p_dir)

    X_new = []
    print(f"\n📸 重新抽取 {person}（{len(imgs)} 張）")

    for img in tqdm(imgs):
        path = f"{p_dir}/{img}"
        img = imread_safe(path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        X_new.append(faces[0].normed_embedding)

    return np.array(X_new)

# ======================================================
# 重新組合 X / y
# ======================================================
def rebuild_dataset(X_old, y_old, label_map_old, added, deleted, changed):
    """
    規則 B（你選的）：
    若某人照片有變 → 重抽該人的 embeddings
    若新增 → 新增 embeddings
    若刪除 → 移除所有舊 embeddings
    """

    # 當前 RAW 資料夾中人物列表
    current_people = sorted(os.listdir(RAW_DIR))

    # 新的 label_map（重新排序）
    new_label_map = {p: i for i, p in enumerate(current_people)}
    new_X = []
    new_y = []

    # --- 對每個現存的人做處理 ---
    for person in current_people:
        if person in added or person in changed:
            # 🔥 必須重抽
            Xp = extract_person_embeddings(person)
        else:
            # 🔥 從舊資料挑出
            if person in label_map_old:
                old_idx = label_map_old[person]
                Xp = X_old[y_old == old_idx]
            else:
                # 理論上不會發生
                Xp = extract_person_embeddings(person)

        # 加入到新集合
        label_idx = new_label_map[person]
        for emb in Xp:
            new_X.append(emb)
            new_y.append(label_idx)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return new_X, new_y, new_label_map

# ======================================================
# 訓練 KNN / SVM / Centers
# ======================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_knn(X, y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def train_svm(X, y):
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X, y)
    return svm

def calc_centers(X, y):
    centers = {}
    for idx in np.unique(y):
        centers[idx] = X[y == idx].mean(axis=0)
    return centers

# ======================================================
# 保存模型
# ======================================================
def save_all(X, y, label_map, knn, svm, centers):
    np.save(f"{MODEL_DIR}/X.npy", X)
    np.save(f"{MODEL_DIR}/y.npy", y)

    json.dump(label_map, open(f"{MODEL_DIR}/label_map.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    pickle.dump(knn, open(f"{MODEL_DIR}/knn.pkl", "wb"))
    pickle.dump(svm, open(f"{MODEL_DIR}/svm.pkl", "wb"))
    pickle.dump(centers, open(f"{MODEL_DIR}/centers.pkl", "wb"))

    print("\n💾 已保存所有模型和資料！")

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    X_old, y_old, label_map_old = load_old_embeddings()
    added, deleted, changed = detect_changes(label_map_old)

    # 若沒有變化 → 不訓練
    if len(added) + len(deleted) + len(changed) == 0:
        print("✔ 沒有變化，不需重新訓練")
        exit()

    print("\n🚀 開始增量訓練（B 版 / 最乾淨模式）")

    X_new, y_new, new_label_map = rebuild_dataset(
        X_old, y_old, label_map_old,
        added, deleted, changed
    )

    print("\n🔧 訓練 KNN / SVM / Centers 中...")
    knn = train_knn(X_new, y_new)
    svm = train_svm(X_new, y_new)
    centers = calc_centers(X_new, y_new)

    save_all(X_new, y_new, new_label_map, knn, svm, centers)

    print("\n🎉 重新訓練完成！")
