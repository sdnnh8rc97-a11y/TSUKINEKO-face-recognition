import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_embedder():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app.models['recognition']   # ArcFaceONNX 物件

def get_embedding(embedder, img, bbox):
    # bbox = [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    face_crop = img[y1:y2, x1:x2]

    # arcface 需要 (112,112)
    face_crop = cv2.resize(face_crop, (112, 112))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # HWC → CHW
    face_crop = face_crop.transpose(2, 0, 1)

    # normalize
    face_crop = (face_crop - 127.5) / 128.0
    face_crop = np.expand_dims(face_crop, axis=0).astype(np.float32)

    # ⭐ 正確的 ArcFace embedding 推論方式
    emb = embedder.get(face_crop)[0]

    # ⭐ L2 normalize
    emb = emb / np.linalg.norm(emb)

    return emb
