"""
Thread reconstruction module for AppleSupport customer support interactions.
Rebuilds multi-turn conversational trees, isolating root customer problems,
agent responses, and complete turn sequences.
"""

from typing import Dict, List, Any, Optional, Generator
import re


def clean_text_for_modeling(text: str) -> str:
    """Normalize text by stripping mention prefixes, redundant URLs, and whitespace."""
    # Replace t.co URLs with URL token or clean spacing
    cleaned = re.sub(r"https?://\S+", "", text)
    # Remove leading customer/agent handles like @AppleSupport @115854
    cleaned = re.sub(r"^(@\w+\s*)+", "", cleaned)
    # Clean redundant spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class ThreadReconstructor:
    """
    Reconstructs conversation trees from a collection of raw tweets.
    """

    def __init__(self):
        self.tweets_by_id: Dict[str, Dict[str, Any]] = {}
        self.children_by_parent: Dict[str, List[str]] = {}

    def add_tweet(self, tweet: Dict[str, Any]) -> None:
        """Add a tweet record to internal lookup tables."""
        tid = tweet["tweet_id"]
        self.tweets_by_id[tid] = tweet
        parent_id = tweet.get("in_response_to_tweet_id", "").strip()
        if parent_id:
            if parent_id not in self.children_by_parent:
                self.children_by_parent[parent_id] = []
            self.children_by_parent[parent_id].append(tid)

    def load_tweets(self, tweets: List[Dict[str, Any]]) -> None:
        """Bulk load tweets."""
        for t in tweets:
            self.add_tweet(t)

    def find_root_inbounds(self) -> List[Dict[str, Any]]:
        """
        Find root inbound tweets from customers directed to AppleSupport.
        Conditions:
        - Inbound is True (or author is not AppleSupport and text mentions @AppleSupport)
        - in_response_to_tweet_id is empty or points to an unknown non-Apple tweet
        """
        roots = []
        for tid, tweet in self.tweets_by_id.items():
            author = tweet.get("author_id", "")
            inbound = tweet.get("inbound", False)
            parent_id = tweet.get("in_response_to_tweet_id", "").strip()
            text = tweet.get("text", "")

            # Customer tweet initiating contact with AppleSupport
            is_customer = author != "AppleSupport" and inbound
            is_root = not parent_id or parent_id not in self.tweets_by_id
            is_for_apple = "@applesupport" in text.lower()

            if is_customer and is_root and is_for_apple:
                roots.append(tweet)
        return roots

    def build_thread(self, root_tweet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build a chronologically ordered conversation thread starting from root_tweet.
        Only keeps threads where AppleSupport replied at least once.
        """
        root_id = root_tweet["tweet_id"]
        turns = []
        visited = set()

        def traverse(curr_id: str):
            if curr_id in visited or curr_id not in self.tweets_by_id:
                return
            visited.add(curr_id)
            t = self.tweets_by_id[curr_id]
            speaker = "agent" if t.get("author_id") == "AppleSupport" else "customer"
            turns.append({
                "tweet_id": curr_id,
                "author_id": t.get("author_id"),
                "speaker": speaker,
                "text": t.get("text", ""),
                "clean_text": clean_text_for_modeling(t.get("text", "")),
                "created_at": t.get("created_at", "")
            })

            # Check direct children
            children = self.children_by_parent.get(curr_id, [])
            for child_id in children:
                traverse(child_id)

            # Also check response_tweet_id string
            resp_str = t.get("response_tweet_id", "")
            if resp_str:
                for resp_id in resp_str.split(","):
                    resp_id = resp_id.strip()
                    if resp_id:
                        traverse(resp_id)

        traverse(root_id)

        # Verify AppleSupport actually participated in this thread
        has_apple_agent = any(turn["speaker"] == "agent" for turn in turns)
        if not has_apple_agent:
            return None

        # Extract agent's first reply
        agent_replies = [turn for turn in turns if turn["speaker"] == "agent"]
        first_agent_reply = agent_replies[0]["text"] if agent_replies else ""

        # Check for presence of actionable resolution / guidance in agent reply
        has_url = "http" in first_agent_reply
        has_dm_link = "dm" in first_agent_reply.lower() or "t.co" in first_agent_reply

        return {
            "conversation_id": f"conv_{root_id}",
            "root_tweet_id": root_id,
            "customer_id": root_tweet.get("author_id"),
            "customer_initial_query": root_tweet.get("text", ""),
            "clean_customer_query": clean_text_for_modeling(root_tweet.get("text", "")),
            "agent_first_reply": first_agent_reply,
            "clean_agent_reply": clean_text_for_modeling(first_agent_reply),
            "num_turns": len(turns),
            "turns": turns,
            "has_link_or_dm": has_dm_link or has_url
        }

    def reconstruct_all_conversations(
        self,
        min_query_chars: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct all resolved conversations from loaded tweets.
        Filters out low-content / empty queries.
        """
        roots = self.find_root_inbounds()
        conversations = []
        for root in roots:
            thread = self.build_thread(root)
            if thread:
                # Discard trivial or non-informative root queries
                clean_q = thread["clean_customer_query"]
                if len(clean_q) >= min_query_chars:
                    conversations.append(thread)
        return conversations
