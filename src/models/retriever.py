"""
Historical Support Retrieval Engine for SupportPilot AI.
Indexes verified, resolved historical AppleSupport conversations and retrieves
the top-K most semantically and lexically relevant historical cases to ground
agent response generation and decision making.

Enforces strict evaluation leakage prevention by rejecting forbidden IDs.
"""

import os
import json
from typing import List, Dict, Any, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class HistoricalSupportRetriever:
    def __init__(
        self,
        top_k: int = 3,
        min_similarity_threshold: float = 0.05,
        ngram_range: tuple = (1, 2),
        max_features: int = 10000
    ):
        self.top_k = top_k
        self.min_similarity_threshold = min_similarity_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )
        self.index_matrix: Optional[np.ndarray] = None
        self.corpus: List[Dict[str, Any]] = []
        self.forbidden_ids: Set[str] = set()

    def build_index(
        self,
        conversations: List[Dict[str, Any]],
        forbidden_ids: Optional[Set[str]] = None
    ) -> "HistoricalSupportRetriever":
        """
        Builds the retrieval index from historical conversations, strictly excluding
        any conversation IDs present in forbidden_ids (evaluation set isolation).
        """
        if forbidden_ids:
            self.forbidden_ids = set(str(fid) for fid in forbidden_ids)

        self.corpus = []
        texts_to_index = []

        for conv in conversations:
            cid = str(conv.get("conversation_id", conv.get("original_conversation_id", "")))
            if cid in self.forbidden_ids:
                continue  # Strict leakage prevention

            # Must have customer query and agent reply
            cust_text = conv.get("clean_customer_query", conv.get("customer_initial_query", "")).strip()
            agent_text = conv.get("agent_first_reply", conv.get("historical_agent_reply", "")).strip()
            if not cust_text or not agent_text:
                continue

            doc = {
                "conversation_id": cid,
                "root_tweet_id": str(conv.get("root_tweet_id", "")),
                "customer_query": cust_text,
                "agent_reply": agent_text,
                "intent": conv.get("ground_truth_intent", conv.get("intent", "general_inquiry_other")),
                "num_turns": conv.get("num_turns", 2)
            }
            self.corpus.append(doc)
            texts_to_index.append(cust_text)

        if not self.corpus:
            raise ValueError("No valid conversations remained to build the retrieval index.")

        self.index_matrix = self.vectorizer.fit_transform(texts_to_index)
        return self

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-K most similar historical conversations for a given customer query.
        """
        if self.index_matrix is None or not self.corpus:
            raise ValueError("Retriever index has not been built yet.")

        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.index_matrix)[0]

        # Get top indices sorted descending
        top_indices = np.argsort(sims)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(sims[idx])
            matched_doc = self.corpus[idx]
            results.append({
                "rank": rank,
                "similarity_score": round(score, 4),
                "conversation_id": matched_doc["conversation_id"],
                "matched_customer_query": matched_doc["customer_query"],
                "historical_agent_reply": matched_doc["agent_reply"],
                "historical_intent": matched_doc["intent"],
                "is_confident_match": score >= self.min_similarity_threshold
            })
        return results

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "index_matrix": self.index_matrix,
            "corpus": self.corpus,
            "forbidden_ids": self.forbidden_ids,
            "top_k": self.top_k,
            "min_similarity_threshold": self.min_similarity_threshold
        }, file_path)

    @classmethod
    def load(cls, file_path: str) -> "HistoricalSupportRetriever":
        data = joblib.load(file_path)
        retriever = cls(
            top_k=data["top_k"],
            min_similarity_threshold=data["min_similarity_threshold"]
        )
        retriever.vectorizer = data["vectorizer"]
        retriever.index_matrix = data["index_matrix"]
        retriever.corpus = data["corpus"]
        retriever.forbidden_ids = data["forbidden_ids"]
        return retriever
