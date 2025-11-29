import pickle
import numpy as np

MODEL_PATH = "models/svm_classifier.pkl"

def load_svm():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def svm_predict(model, emb):
    pred = model.predict([emb])[0]
    conf = max(model.predict_proba([emb])[0])
    return pred, float(conf)
