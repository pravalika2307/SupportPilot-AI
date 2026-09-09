"""
TF-IDF + Logistic Regression Intent Classification Baseline for SupportPilot AI.
Extracts unigram and bigram TF-IDF representations and trains a regularized,
class-weighted Logistic Regression classifier.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


class TfidfLogisticRegressionBaseline:
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        C: float = 1.0,
        random_state: int = 42
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
            solver="lbfgs"
        )
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "TfidfLogisticRegressionBaseline":
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier)
        ])
        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, X: List[str]) -> List[str]:
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet.")
        return list(self.pipeline.predict(X))

    def predict_proba(self, X: List[str]) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet.")
        return self.pipeline.predict_proba(X)

    def predict_with_confidence(self, X: List[str]) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet.")
        probas = self.predict_proba(X)
        results = []
        for proba_row in probas:
            max_idx = int(np.argmax(proba_row))
            intent = self.classes_[max_idx]
            conf = float(proba_row[max_idx])
            prob_dict = {cls_name: float(p) for cls_name, p in zip(self.classes_, proba_row)}
            results.append({
                "intent": intent,
                "confidence": conf,
                "probabilities": prob_dict
            })
        return results

    def get_top_features_per_intent(self, top_n: int = 8) -> Dict[str, List[Tuple[str, float]]]:
        """Extract top positive n-grams influencing each intent prediction."""
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet.")
        vec = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feature_names = vec.get_feature_names_out()

        top_features = {}
        for idx, cls_name in enumerate(self.classes_):
            coefs = clf.coef_[idx]
            top_indices = np.argsort(coefs)[::-1][:top_n]
            top_features[cls_name] = [
                (str(feature_names[i]), float(coefs[i])) for i in top_indices
            ]
        return top_features

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "classes_": self.classes_,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "C": self.C,
            "random_state": self.random_state
        }, file_path)

    @classmethod
    def load(cls, file_path: str) -> "TfidfLogisticRegressionBaseline":
        data = joblib.load(file_path)
        instance = cls(
            ngram_range=data["ngram_range"],
            max_features=data["max_features"],
            C=data["C"],
            random_state=data["random_state"]
        )
        instance.pipeline = data["pipeline"]
        instance.classes_ = data["classes_"]
        return instance
