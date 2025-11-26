import cv2
import numpy as np
from face_detector import FaceDetector
from face_embedder import FaceEmbedder
from face_classifier import FaceClassifier
import joblib
import os

det = FaceDetector()
embedder = FaceEmbedder()

# 載入模型
label_map = joblib.load("models/label_map.pkl")
knn = joblib.load("models/knn.pkl")
svm = joblib.load("models/svm.pkl")
centers = joblib.load("models/centers.pkl")

clf = FaceClassifier(label_map, knn, svm, centers)

def recognize_group(image_path, output_path):
    img = cv2.imread(image_path)

    faces = det.detect_faces(img)

    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        face_crop = det.crop_face(img, f["bbox"])

        emb = embedder.get_embedding(face_crop)

        name = "Unknown"
        if emb is not None:
            name = clf.vote(emb)

        # 畫框
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite(output_path, img)
    print("輸出完成:", output_path)
