import numpy as np
import pickle
import os

MODEL_PATH = "models/cosine_centers.pkl"

def cosine_predict(embedding):
    with open(MODEL_PATH, "rb") as f:
        centers = pickle.load(f)

    best_label = "Unknown"
    best_score = -1

    for label, center in centers.items():
        center = np.array(center)
        score = np.dot(embedding, center)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, float(best_score)
