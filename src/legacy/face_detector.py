import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.detector = FaceAnalysis(name="buffalo_l")
        self.detector.prepare(ctx_id=0, det_size=(1280, 1280))

    def detect_faces(self, image):
        """偵測所有臉，回傳 (bbox, landmark, score)"""
        faces = self.detector.get(image)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "score": face.det_score,
                "embedding": face.normed_embedding if hasattr(face, "normed_embedding") else None
            })
        return results

    def crop_face(self, image, bbox, margin=20):
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        x1 = max(x1 - margin, 0)
        y1 = max(y1 - margin, 0)
        x2 = min(x2 + margin, w)
        y2 = min(y2 + margin, h)

        return image[y1:y2, x1:x2]
