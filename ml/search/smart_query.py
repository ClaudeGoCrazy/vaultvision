"""
Claude-Powered Smart NL Query Engine
Uses Claude API to interpret complex natural language queries into
structured filters + ChromaDB search. Falls back to direct ChromaDB.
"""
import json
import logging
from typing import Optional

from ml.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from ml.search.query_engine import QueryEngine

logger = logging.getLogger(__name__)


class SmartQueryEngine:
    """
    Wraps QueryEngine with Claude-powered query interpretation.
    Handles complex queries like:
      - "show me suspicious activity after 5pm"
      - "when did the red truck leave"
      - "any loitering near the entrance"
      - "count all people who entered from the left"
    """

    def __init__(self):
        self.query_engine = QueryEngine()

    def search(
        self,
        query: str,
        video_id: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """
        Smart search: interprets query with Claude, then searches ChromaDB.

        Returns:
            {
                "original_query": str,
                "interpreted_query": str | None,
                "filters": dict | None,
                "results": list[dict],
                "reasoning": str | None,
            }
        """
        interpreted = None
        filters = None
        reasoning = None

        if ANTHROPIC_API_KEY:
            try:
                interpreted, filters, reasoning = self._interpret_query(query)
            except Exception as e:
                logger.warning(f"Claude query interpretation failed: {e}")

        # Use interpreted query for ChromaDB search, or fall back to original
        search_query = interpreted or query
        results = self.query_engine.search(
            query=search_query,
            video_id=video_id,
            limit=limit,
        )

        # Apply any structured filters from Claude
        if filters:
            results = self._apply_filters(results, filters)

        return {
            "original_query": query,
            "interpreted_query": interpreted,
            "filters": filters,
            "results": results,
            "reasoning": reasoning,
        }

    def _interpret_query(self, query: str) -> tuple[str | None, dict | None, str | None]:
        """Use Claude to interpret a complex NL query."""
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""You are a video surveillance query interpreter. Given a user's natural language query about surveillance footage, extract:

1. A simplified search query optimized for semantic similarity search (for ChromaDB)
2. Any structured filters that should be applied

User query: "{query}"

Respond in this exact JSON format:
{{
  "search_query": "simplified query for semantic search",
  "filters": {{
    "event_types": ["entry", "exit", "loitering", "zone_intrusion", "crowd_threshold", "object_left"] or null,
    "class_names": ["person", "car", "truck", "bus", etc.] or null,
    "time_after_sec": number or null,
    "time_before_sec": number or null
  }},
  "reasoning": "brief explanation of interpretation"
}}

Only include filters that are clearly indicated by the query. Set unused filters to null."""

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Parse JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        search_query = parsed.get("search_query")
        filters = parsed.get("filters")
        reasoning = parsed.get("reasoning")

        # Clean null filters
        if filters:
            filters = {k: v for k, v in filters.items() if v is not None}

        logger.info(f"Query interpreted: '{query}' -> '{search_query}' (filters: {filters})")
        return search_query, filters if filters else None, reasoning

    def _apply_filters(self, results: list[dict], filters: dict) -> list[dict]:
        """Apply structured filters to search results."""
        filtered = results

        event_types = filters.get("event_types")
        if event_types:
            filtered = [r for r in filtered if r.get("event_type") in event_types]

        class_names = filters.get("class_names")
        if class_names:
            filtered = [r for r in filtered if r.get("class_name") in class_names]

        time_after = filters.get("time_after_sec")
        if time_after is not None:
            filtered = [r for r in filtered if r.get("start_time_sec", 0) >= time_after]

        time_before = filters.get("time_before_sec")
        if time_before is not None:
            filtered = [r for r in filtered if r.get("start_time_sec", float("inf")) <= time_before]

        return filtered

    def index_events(self, events: list[dict], video_id: str, video_filename: str = ""):
        """Delegate to underlying QueryEngine."""
        self.query_engine.index_events(events, video_id, video_filename)

    def delete_video_events(self, video_id: str):
        """Delegate to underlying QueryEngine."""
        self.query_engine.delete_video_events(video_id)
