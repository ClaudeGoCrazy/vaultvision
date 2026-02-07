"""
Natural Language Query Engine
Embeds event descriptions into ChromaDB and enables NL search via Claude API.
"""
import logging
from typing import Optional

import chromadb

from ml.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Manages event embeddings in ChromaDB and provides natural language search.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized with collection '{CHROMA_COLLECTION_NAME}'")

    def index_events(self, events: list[dict], video_id: str, video_filename: str = ""):
        """
        Index event descriptions into ChromaDB for later search.

        Args:
            events: List of Event dicts with descriptions
            video_id: Video ID to associate events with
            video_filename: Original filename for search results
        """
        if not events:
            logger.warning("No events to index")
            return

        ids = []
        documents = []
        metadatas = []

        for event in events:
            event_id = event["event_id"]
            ids.append(f"{video_id}_{event_id}")
            documents.append(event["description"])
            metadatas.append({
                "video_id": video_id,
                "video_filename": video_filename,
                "event_id": event_id,
                "event_type": event["event_type"],
                "class_name": event["class_name"],
                "start_time_sec": event["start_time_sec"],
                "end_time_sec": event.get("end_time_sec") or -1,
                "confidence": event["confidence"],
                "track_id": event.get("track_id") or -1,
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(events)} events for video {video_id}")

    def search(
        self,
        query: str,
        video_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search events using natural language query.

        Args:
            query: Natural language query string
            video_id: Optional video ID to scope search
            limit: Maximum results to return

        Returns:
            List of search result dicts with event metadata and relevance scores
        """
        where_filter = None
        if video_id:
            where_filter = {"video_id": video_id}

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter,
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                # Convert distance to relevance score (cosine distance -> similarity)
                relevance = max(0.0, min(1.0, 1.0 - distance))

                search_results.append({
                    "event_id": metadata.get("event_id", ""),
                    "video_id": metadata.get("video_id", ""),
                    "video_filename": metadata.get("video_filename", ""),
                    "event_type": metadata.get("event_type", ""),
                    "class_name": metadata.get("class_name", ""),
                    "description": results["documents"][0][i] if results["documents"] else "",
                    "start_time_sec": metadata.get("start_time_sec", 0),
                    "end_time_sec": metadata.get("end_time_sec", -1),
                    "confidence": metadata.get("confidence", 0),
                    "track_id": metadata.get("track_id", -1),
                    "relevance_score": round(relevance, 4),
                })

        # Sort by relevance
        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"Query '{query}' returned {len(search_results)} results")
        return search_results

    def delete_video_events(self, video_id: str):
        """Remove all indexed events for a video."""
        # ChromaDB doesn't support delete by metadata directly in all versions,
        # so we query first then delete by IDs
        results = self.collection.get(
            where={"video_id": video_id},
        )
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} indexed events for video {video_id}")

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "total_events_indexed": self.collection.count(),
            "collection_name": CHROMA_COLLECTION_NAME,
        }
