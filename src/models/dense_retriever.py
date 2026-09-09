"""
Dense Semantic Support Retriever for SupportPilot AI.
Uses SentenceTransformers (all-MiniLM-L6-v2) to map customer support queries
and historical AppleSupport interactions into 384-dimensional dense semantic vectors.

Computes cosine similarity via dot product over L2-normalized embeddings,
enabling robust semantic matching beyond sparse lexical overlap.
Enforces strict evaluation leakage prevention by excluding forbidden IDs.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Set
import numpy as np
import joblib

# Lazy import or direct import
from sentence_transformers import SentenceTransformer


class DenseSupportRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        min_similarity_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.min_similarity_threshold = min_similarity_threshold
        self.device = device
        self.model: Optional[SentenceTransformer] = None
        self.corpus: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.forbidden_ids: Set[str] = set()

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def build_index(
        self,
        conversations: List[Dict[str, Any]],
        forbidden_ids: Optional[Set[str]] = None,
        batch_size: int = 64
    ) -> "DenseSupportRetriever":
        """
        Encodes historical conversations into dense vectors, strictly filtering out
        any conversation IDs present in forbidden_ids (zero evaluation leakage).
        """
        if forbidden_ids:
            self.forbidden_ids = set(str(fid) for fid in forbidden_ids)

        self.corpus = []
        texts_to_encode = []

        for conv in conversations:
            cid = str(conv.get("conversation_id", conv.get("original_conversation_id", "")))
            if cid in self.forbidden_ids:
                continue  # Strict leakage prevention

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
            texts_to_encode.append(cust_text)

        if not self.corpus:
            raise ValueError("No valid conversations remained to build dense retrieval index.")

        model = self._load_model()
        raw_embeddings = model.encode(
            texts_to_encode,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        self.embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        return self

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Encodes the query into a normalized dense vector and retrieves the top-K
        most similar historical conversations via dot product.
        """
        if self.embeddings is None or not self.corpus:
            raise ValueError("Dense retriever index has not been built yet.")

        k = top_k or self.top_k
        model = self._load_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_emb[0], dtype=np.float32)

        # Dot product with normalized embeddings equals cosine similarity
        sims = np.dot(self.embeddings, query_vec)
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
        """Saves corpus metadata, precomputed embeddings, and config to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump({
            "model_name": self.model_name,
            "embeddings": self.embeddings,
            "corpus": self.corpus,
            "forbidden_ids": self.forbidden_ids,
            "top_k": self.top_k,
            "min_similarity_threshold": self.min_similarity_threshold
        }, file_path)

    @classmethod
    def load(cls, file_path: str, device: Optional[str] = None) -> "DenseSupportRetriever":
        """Loads precomputed dense index and initializes SentenceTransformer for inference."""
        data = joblib.load(file_path)
        instance = cls(
            model_name=data["model_name"],
            top_k=data["top_k"],
            min_similarity_threshold=data["min_similarity_threshold"],
            device=device
        )
        instance.embeddings = data["embeddings"]
        instance.corpus = data["corpus"]
        instance.forbidden_ids = data["forbidden_ids"]
        return instance
