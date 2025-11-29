import cv2
import numpy as np

def load_embedder():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app.models['recognition']


def get_embedding(embedder, img_bytes, bbox):
    # --- 將 bytes 轉成 numpy array ---
    if isinstance(img_bytes, bytes):
        img_bytes = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imdecode failed: image bytes invalid")

    # --- 依照 bbox 裁切 ---
    x1, y1, x2, y2 = bbox
    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise ValueError("Empty face crop from bbox")

    # --- InsightFace 要求輸入 112x112 RGB ---
    face_crop = cv2.resize(face_crop, (112, 112))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = np.expand_dims(face_crop.transpose(2, 0, 1), axis=0)
    face_crop = (face_crop - 127.5) / 128.0

    # --- 取 embedding ---
    emb = embedder(face_crop)[0]
    emb = emb / np.linalg.norm(emb)

    return emb
