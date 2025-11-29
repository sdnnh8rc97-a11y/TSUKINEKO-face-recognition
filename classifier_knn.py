import pickle
import numpy as np

MODEL_PATH = "models/knn_classifier.pkl"

def load_knn():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def knn_predict(model, emb):
    dist, idx = model.kneighbors([emb], n_neighbors=1)
    dist = dist[0][0]
    pred = model.predict([emb])[0]
    score = max(0, 1 - dist)
    return pred, float(score)
