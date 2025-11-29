import cv2
import numpy as np

def load_embedder():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app.models['recognition']

def get_embedding(embedder, img_bytes, bbox):
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = bbox
    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (112, 112))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = face_crop.transpose(2, 0, 1)
    face_crop = np.expand_dims(face_crop, axis=0).astype(np.float32)
    face_crop = (face_crop - 127.5) / 128.0
    emb = embedder(face_crop)[0]
    return emb / np.linalg.norm(emb)
