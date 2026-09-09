"""
Majority-Class Intent Classification Baseline for SupportPilot AI.
Predicts the most frequent class observed in the training distribution.
Serves as the zero-intelligence lower bound benchmark.
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


class MajorityClassBaseline:
    def __init__(self):
        self.majority_class: Optional[str] = None
        self.class_priors: Dict[str, float] = {}
        self.classes_: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "MajorityClassBaseline":
        counts = Counter(y)
        total = len(y)
        self.classes_ = sorted(counts.keys())
        self.class_priors = {c: counts[c] / total for c in self.classes_}
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X: List[str]) -> List[str]:
        if self.majority_class is None:
            raise ValueError("Baseline has not been fitted yet.")
        return [self.majority_class for _ in X]

    def predict_proba(self, X: List[str]) -> np.ndarray:
        if self.majority_class is None:
            raise ValueError("Baseline has not been fitted yet.")
        row = [self.class_priors[c] for c in self.classes_]
        return np.tile(row, (len(X), 1))

    def predict_with_confidence(self, X: List[str]) -> List[Dict[str, Any]]:
        preds = self.predict(X)
        conf = self.class_priors[self.majority_class]
        return [{"intent": p, "confidence": conf} for p in preds]
