import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import Video, Event

router = APIRouter(prefix="/api/v1", tags=["query"])

# Synonym expansion — maps user-friendly terms to words found in event descriptions / class names
SYNONYMS = {
    "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
    "car": ["car", "vehicle"],
    "truck": ["truck", "vehicle"],
    "bus": ["bus", "vehicle"],
    "person": ["person", "people", "someone", "individual", "pedestrian", "man", "woman"],
    "people": ["person", "people", "someone", "pedestrian"],
    "someone": ["person", "someone", "individual"],
    "walking": ["entered", "exited", "walked", "walking", "visible", "moved"],
    "moving": ["entered", "exited", "moved", "moving", "visible"],
    "entered": ["entered", "entry", "appeared"],
    "left": ["exited", "exit", "left", "departed"],
    "exited": ["exited", "exit", "left"],
    "loitering": ["loitering", "stationary", "lingered", "stayed"],
    "crowd": ["crowd", "group", "gathered", "multiple"],
    "suspicious": ["anomaly", "loitering", "intrusion", "abandoned"],
    "bag": ["backpack", "handbag", "suitcase", "bag"],
    "animal": ["dog", "cat", "animal"],
}


def expand_query(query_text: str) -> list[str]:
    """Expand query words with synonyms for broader matching."""
    words = query_text.lower().split()
    expanded = set(words)
    for w in words:
        if w in SYNONYMS:
            expanded.update(SYNONYMS[w])
    return list(expanded)


def score_event(event_desc: str, event_class: str, event_type: str, query_words: list[str]) -> float:
    """Score an event against expanded query words. Returns 0.0-1.0."""
    searchable = f"{event_desc} {event_class} {event_type}".lower()
    matches = sum(1 for w in query_words if w in searchable)
    if matches == 0:
        return 0.0
    return min(matches / max(len(query_words), 1), 1.0)


@router.post("/query")
async def natural_language_query(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    query_text = body.get("query", "")
    video_id = body.get("video_id")
    limit = min(body.get("limit", 10), 50)

    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")

    start = time.perf_counter()

    # Build base query
    stmt = select(Event)
    if video_id:
        stmt = stmt.where(Event.video_id == video_id)

    result = await db.execute(stmt.order_by(Event.start_time_sec))
    all_events = result.scalars().all()

    # Expand query with synonyms
    expanded_words = expand_query(query_text)

    scored_results = []
    for event in all_events:
        relevance = score_event(
            event.description, event.class_name, event.event_type, expanded_words
        )
        if relevance > 0:
            scored_results.append((event, relevance))

    # Sort by relevance descending
    scored_results.sort(key=lambda x: x[1], reverse=True)
    scored_results = scored_results[:limit]

    # Build response with video info
    results = []
    for event, relevance in scored_results:
        video = await db.get(Video, event.video_id)
        results.append({
            "event": {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "class_name": event.class_name,
                "track_id": event.track_id,
                "start_time_sec": event.start_time_sec,
                "end_time_sec": event.end_time_sec,
                "description": event.description,
                "confidence": event.confidence,
                "metadata": event.metadata_json or {},
            },
            "video_id": event.video_id,
            "video_filename": video.original_filename if video else "unknown",
            "relevance_score": round(relevance, 3),
            "thumbnail_path": video.thumbnail_path if video else None,
        })

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "query": query_text,
        "results": results,
        "total_results": len(results),
        "processing_time_ms": round(elapsed_ms, 2),
    }
