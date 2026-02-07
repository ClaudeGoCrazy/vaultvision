"""
Claude-Powered Video Activity Summary
Generates a natural language summary of all detected activity in a video.
Falls back to a template-based summary if no API key is configured.
"""
import logging
from collections import Counter

from ml.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _template_summary(
    events: list[dict],
    track_summaries: dict,
    class_counts: dict,
    video_duration_sec: float,
    unique_persons: int,
    unique_vehicles: int,
) -> str:
    """Generate a template-based summary without Claude API."""
    parts = []

    # Duration
    parts.append(f"In this {_format_duration(video_duration_sec)} video:")

    # People
    if unique_persons > 0:
        parts.append(f"- {unique_persons} unique person(s) detected")

    # Vehicles
    if unique_vehicles > 0:
        vehicle_types = []
        for cls in ["car", "truck", "bus", "motorcycle", "bicycle"]:
            if cls in class_counts:
                vehicle_types.append(f"{cls}s")
        vtype_str = ", ".join(vehicle_types) if vehicle_types else "vehicles"
        parts.append(f"- {unique_vehicles} unique vehicle(s) detected ({vtype_str})")

    # Other objects
    other_classes = {k: v for k, v in class_counts.items()
                     if k not in ("person", "car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "other")}
    for cls, count in other_classes.items():
        parts.append(f"- {count} {cls} detection(s)")

    # Events breakdown
    event_types = Counter(e["event_type"] for e in events)
    if event_types.get("loitering"):
        parts.append(f"- {event_types['loitering']} loitering incident(s)")
    if event_types.get("zone_intrusion"):
        parts.append(f"- {event_types['zone_intrusion']} zone intrusion(s)")
    if event_types.get("crowd_threshold"):
        parts.append(f"- {event_types['crowd_threshold']} crowd event(s)")
    if event_types.get("object_left"):
        parts.append(f"- {event_types['object_left']} abandoned object(s)")

    # Track durations
    durations = []
    for tid, summary in track_summaries.items():
        dur = summary["last_seen_sec"] - summary["first_seen_sec"]
        if dur > 0:
            durations.append(dur)
    if durations:
        avg_dur = sum(durations) / len(durations)
        max_dur = max(durations)
        parts.append(f"- Average track duration: {_format_duration(avg_dur)}, longest: {_format_duration(max_dur)}")

    return "\n".join(parts)


def generate_summary(
    events: list[dict],
    track_summaries: dict,
    class_counts: dict,
    video_duration_sec: float,
    unique_persons: int,
    unique_vehicles: int,
    video_filename: str = "",
) -> str:
    """
    Generate a natural language summary of video activity.
    Uses Claude API if available, falls back to template.

    Returns:
        Human-readable summary string
    """
    # Always generate template first as fallback
    template = _template_summary(
        events, track_summaries, class_counts,
        video_duration_sec, unique_persons, unique_vehicles,
    )

    if not ANTHROPIC_API_KEY:
        logger.info("No Anthropic API key — using template summary")
        return template

    # Try Claude API for a richer summary
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Build context for Claude
        event_descriptions = [e["description"] for e in events[:50]]  # cap at 50

        prompt = f"""You are an AI video analyst. Summarize the following surveillance video analysis results in 3-5 concise sentences.

Video: {video_filename or 'Unknown'}
Duration: {_format_duration(video_duration_sec)}
Unique persons detected: {unique_persons}
Unique vehicles detected: {unique_vehicles}
Object counts: {class_counts}

Events detected:
{chr(10).join(f'- {desc}' for desc in event_descriptions)}

Write a professional surveillance summary. Be specific about what was observed, when, and any notable patterns or anomalies. Do not speculate beyond what the data shows."""

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.content[0].text.strip()
        logger.info("Generated Claude-powered video summary")
        return summary

    except Exception as e:
        logger.warning(f"Claude summary failed, using template: {e}")
        return template
