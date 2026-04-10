import math
import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
else:
    print("[WARNING] OPENAI_API_KEY not set (embeddings)")
    client = None
    
# simple cache
_embedding_cache = {}

def safe_embedding(text: str):
    if client is None:
        return None

    try:
        if not text or not text.strip():
            return None

        if text in _embedding_cache:
            return _embedding_cache[text]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        emb = response.data[0].embedding
        _embedding_cache[text] = emb
        return emb

    except Exception as e:
        print("[EMBED ERROR]", e)
        return None


def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0

    dot = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)