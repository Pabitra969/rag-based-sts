# retrievers/base_retriever.py
from typing import List, Dict, Any
import abc

class BaseRetriever(abc.ABC):
    """
    Standard interface for all retrievers. The main mind
    Subclasses must implement `search`.
    `search` returns a list of dicts:
      [{"source": "chroma", "content": "<text>", "metadata": {...}, "score": 0.92}, ...]
    """
    @abc.abstractmethod
    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError
