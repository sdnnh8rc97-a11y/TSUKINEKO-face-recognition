import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from numpy.linalg import norm

class FaceClassifier:
    def __init__(self, label_map, knn_model, svm_model, centers):
        self.label_map = label_map
        self.knn = knn_model
        self.svm = svm_model
        self.centers = centers  # { "Momona": vector, ... }

    def cosine_predict(self, emb):
        best_name = "Unknown"
        best_score = -1

        for name, center in self.centers.items():
            sim = np.dot(emb, center) / (norm(emb) * norm(center))
            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score < 0.45:   # 門檻可調
            return "Unknown"

        return best_name

    def vote(self, emb):
        """Cosine + KNN + SVM 多模型投票"""

        res_cos = self.cosine_predict(emb)
        res_knn = self.knn.predict([emb])[0]
        res_svm = self.svm.predict([emb])[0]

        votes = [res_cos, res_knn, res_svm]
        result = max(set(votes), key=votes.count)

        # Unknown 門檻：若三票有兩票是 Unknown → Unknown
        if votes.count("Unknown") >= 2:
            return "Unknown"

        return result
