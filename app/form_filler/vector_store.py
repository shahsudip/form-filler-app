import math
import json
import os
import requests
from collections import Counter

def get_char_ngrams(text: str, n: int = 3) -> list:
    """Helper to generate character n-grams for fallback string similarity."""
    text_clean = text.lower().strip()
    if len(text_clean) < n:
        return [text_clean]
    return [text_clean[i:i+n] for i in range(len(text_clean) - n + 1)]

def fallback_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates bag-of-ngrams cosine similarity as a lightweight fallback."""
    vec1 = Counter(get_char_ngrams(text1))
    vec2 = Counter(get_char_ngrams(text2))
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

class SimpleVectorStore:
    def __init__(self, model_name: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.items = []  # List of dicts: {"id": str, "text": str, "metadata": dict, "embedding": list}
        self.use_fallback = False

    def _get_embedding(self, text: str) -> list:
        """Fetches vector embedding from Ollama or returns None if offline."""
        if self.use_fallback:
            return None
        try:
            # Try /api/embed (batch API)
            res = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model_name, "input": text},
                timeout=2.0
            )
            if res.status_code == 200:
                data = res.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]
            
            # Fallback to /api/embeddings (single API)
            res = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=2.0
            )
            if res.status_code == 200:
                data = res.json()
                if "embedding" in data:
                    return data["embedding"]
        except Exception as e:
            # Print ASCII message warning that we are falling back
            print(f"Ollama connection failed, using local n-gram fallback: {str(e)}")
            self.use_fallback = True
            
        return None

    def add_item(self, item_id: str, text: str, metadata: dict = None):
        """Vectorizes and indexes a text chunk with associated metadata."""
        embedding = self._get_embedding(text)
        self.items.append({
            "id": item_id,
            "text": text,
            "metadata": metadata or {},
            "embedding": embedding
        })

    def search(self, query: str, limit: int = 5) -> list:
        """Searches vector store for items similar to the query."""
        if not self.items:
            return []

        query_emb = self._get_embedding(query)
        results = []

        # If query embedding failed or we are using fallback, calculate string similarity
        if self.use_fallback or query_emb is None:
            for item in self.items:
                score = fallback_cosine_similarity(query, item["text"])
                results.append((score, item))
        else:
            for item in self.items:
                item_emb = item["embedding"]
                if item_emb is None:
                    # Calculate string similarity as fallback for items without embedding
                    score = fallback_cosine_similarity(query, item["text"])
                else:
                    # Calculate vector cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_emb, item_emb))
                    norm_a = math.sqrt(sum(a * a for a in query_emb))
                    norm_b = math.sqrt(sum(b * b for b in item_emb))
                    if norm_a == 0 or norm_b == 0:
                        score = 0.0
                    else:
                        score = dot_product / (norm_a * norm_b)
                results.append((score, item))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Return formatted matches
        formatted = []
        for score, item in results[:limit]:
            formatted.append({
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": score
            })
        return formatted


# ---------------------------------------------------------------------------
# ProfileVectorStore
# Manages a persistent user profile (key -> value pairs) and provides
# semantic search to auto-suggest values for detected form field contexts.
# ---------------------------------------------------------------------------

PROFILE_FILENAME = "_user_profile.json"

# Minimum cosine similarity score to accept a suggestion (0.0 - 1.0).
# Below this threshold, no suggestion is returned for a field.
SUGGESTION_THRESHOLD = 0.45


class ProfileVectorStore:
    """
    Stores a user's profile as labeled key-value pairs and enables
    semantic matching of form field context labels to profile keys using
    vector embeddings (Ollama nomic-embed-text) with n-gram fallback.
    """

    def __init__(
        self,
        profile_dir: str,
        embed_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ):
        self.profile_path = os.path.join(profile_dir, PROFILE_FILENAME)
        self._store = SimpleVectorStore(model_name=embed_model, ollama_url=ollama_url)
        # Raw profile data: list of {"key": str, "value": str}
        self._profile: list = []
        self._indexed = False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def load(self) -> list:
        """Loads the stored profile from disk. Returns list of {"key", "value"} dicts."""
        if not os.path.exists(self.profile_path):
            self._profile = []
            return []
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept either a list of {key, value} or a flat dict
            if isinstance(data, list):
                self._profile = data
            elif isinstance(data, dict):
                self._profile = [{"key": k, "value": v} for k, v in data.items()]
            else:
                self._profile = []
        except Exception:
            self._profile = []
        self._indexed = False
        return self._profile

    def save(self, entries: list) -> list:
        """
        Saves profile entries to disk and rebuilds the index.
        entries: list of {"key": str, "value": str}
        Returns the saved entries.
        """
        # Deduplicate by key (last write wins)
        seen = {}
        for entry in entries:
            key = entry.get("key", "").strip()
            value = entry.get("value", "").strip()
            if key:
                seen[key] = value
        self._profile = [{"key": k, "value": v} for k, v in seen.items()]
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self._profile, f, ensure_ascii=False, indent=2)
        self._indexed = False
        return self._profile

    # ------------------------------------------------------------------
    # Vector index management
    # ------------------------------------------------------------------

    def _build_index(self):
        """Rebuilds the vector index from the current in-memory profile."""
        self._store = SimpleVectorStore(
            model_name=self._store.model_name,
            ollama_url=self._store.ollama_url,
        )
        for entry in self._profile:
            key = entry.get("key", "").strip()
            value = entry.get("value", "").strip()
            if key:
                self._store.add_item(
                    item_id=key,
                    text=key,  # We embed the profile KEY (the label) for matching
                    metadata={"value": value},
                )
        self._indexed = True

    def _ensure_indexed(self):
        if not self._indexed:
            self._build_index()

    # ------------------------------------------------------------------
    # Suggestion API
    # ------------------------------------------------------------------

    def suggest(self, field_contexts: list) -> list:
        """
        Given a list of field context strings (detected labels from the PDF),
        returns the best-matching profile value for each field.

        Parameters
        ----------
        field_contexts : list of {"field_id": str, "context": str}

        Returns
        -------
        list of {
            "field_id": str,
            "suggested_value": str,   # empty string if no match
            "matched_key": str,       # the profile key that matched
            "score": float            # similarity score 0.0 - 1.0
        }
        """
        if not self._profile:
            self.load()

        if not self._profile:
            # No profile data, return empty suggestions
            return [
                {
                    "field_id": fc.get("field_id", ""),
                    "suggested_value": "",
                    "matched_key": "",
                    "score": 0.0,
                }
                for fc in field_contexts
            ]

        self._ensure_indexed()

        results = []
        for fc in field_contexts:
            field_id = fc.get("field_id", "")
            context = fc.get("context", "").strip()

            if not context:
                results.append({
                    "field_id": field_id,
                    "suggested_value": "",
                    "matched_key": "",
                    "score": 0.0,
                })
                continue

            matches = self._store.search(context, limit=1)
            if matches and matches[0]["score"] >= SUGGESTION_THRESHOLD:
                best = matches[0]
                results.append({
                    "field_id": field_id,
                    "suggested_value": best["metadata"].get("value", ""),
                    "matched_key": best["id"],
                    "score": round(best["score"], 4),
                })
            else:
                results.append({
                    "field_id": field_id,
                    "suggested_value": "",
                    "matched_key": "",
                    "score": round(matches[0]["score"], 4) if matches else 0.0,
                })

        return results
