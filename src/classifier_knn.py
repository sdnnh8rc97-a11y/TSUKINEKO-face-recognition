import pickle
import os

DEFAULT_KNN_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "knn.pkl"
)

def load_knn(path: str = DEFAULT_KNN_PATH):
    """Load KNN model from given path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def knn_predict(model, emb):
    pred = model.predict([emb])[0]
    score = -model.kneighbors([emb])[0][0][0]  # 距離越小越相似
    return pred, float(score)

