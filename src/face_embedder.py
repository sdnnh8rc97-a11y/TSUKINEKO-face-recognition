import numpy as np
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.model = FaceAnalysis(name="buffalo_l")
        self.model.prepare(ctx_id=0)

    def get_embedding(self, face_img):
        """回傳 512 維 embedding"""
        faces = self.model.get(face_img)
        if len(faces) == 0:
            return None
        return faces[0].normed_embedding
