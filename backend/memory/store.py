"""ChromaDB Memory Store (FR-MAP-11, FR-MEM-01).

Stores approved and rejected mappings as positive/negative examples in a persistent local
ChromaDB vector collection (`data/chroma_db`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    Settings = None
    HAS_CHROMADB = False

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("data/chroma_db")


class MemoryStore:
    def __init__(self, db_dir: Path | str = CHROMA_PATH) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        if HAS_CHROMADB:
            self.client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name="sov_mapping_memory",
                metadata={"description": "Approved and rejected SOV column mappings"},
            )
        else:
            self.client = None
            self.collection = None
            logger.warning("chromadb is not installed. Vector memory store operating in fallback no-op mode.")

    def record_decision(
        self,
        source_column: str,
        target_field: str,
        approved: bool,
        file_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Store a human decision into vector memory."""
        doc_id = f"{'pos' if approved else 'neg'}_{source_column.lower()}_{target_field}_{hash(source_column + target_field)}"
        if not HAS_CHROMADB or self.collection is None:
            return doc_id

        doc_text = f"Source Header: '{source_column}' -> Target Field: '{target_field}' ({'Approved' if approved else 'Rejected'})"

        metadata = {
            "source_column": source_column,
            "target_field": target_field,
            "approved": approved,
            "file_name": file_name or "",
            "notes": notes or "",
        }

        self.collection.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[metadata],
        )
        return doc_id

    def query_similar(self, header: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find past approved/rejected mapping decisions similar to header."""
        if not HAS_CHROMADB or self.collection is None:
            return []
        results = self.collection.query(
            query_texts=[header],
            n_results=min(limit, self.collection.count() or 1),
        )
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return []

        out = []
        for i, meta in enumerate(results["metadatas"][0]):
            dist = results["distances"][0][i] if results.get("distances") else 0.0
            out.append({
                "source_column": meta.get("source_column"),
                "target_field": meta.get("target_field"),
                "approved": meta.get("approved"),
                "distance": dist,
            })
        return out


_STORE: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore()
    return _STORE
