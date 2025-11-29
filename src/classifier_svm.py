import pickle
import os

DEFAULT_SVM_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "svm.pkl"
)

def load_svm(path: str = DEFAULT_SVM_PATH):
    """Load SVM model from given path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def svm_predict(model, emb):
    pred = model.predict([emb])[0]
    score = model.decision_function([emb])[0]
    return pred, float(score)

