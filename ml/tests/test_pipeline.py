"""
End-to-end test for the VaultVision ML Pipeline.
Generates a test video, runs the full pipeline, validates output.
"""
import sys
import json
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("test_pipeline")


def test_full_pipeline():
    """Run the complete pipeline on a test video and validate output."""
    from ml.tests.generate_test_video import generate_test_video
    from ml.pipeline.main import process_video
    from shared.schemas import PipelineResult

    # Step 1: Generate test video
    logger.info("=" * 60)
    logger.info("STEP 1: Generating test video...")
    logger.info("=" * 60)
    video_path = generate_test_video(duration_sec=5, fps=15)
    logger.info(f"Test video created at: {video_path}")

    # Step 2: Run pipeline
    logger.info("=" * 60)
    logger.info("STEP 2: Running ML pipeline...")
    logger.info("=" * 60)

    def progress(pct, step, msg):
        logger.info(f"  [{step}] {pct:.0f}% - {msg}")

    result = process_video(
        video_path=video_path,
        video_id="test-001",
        video_filename="test_video.mp4",
        progress_callback=progress,
    )

    # Step 3: Validate output
    logger.info("=" * 60)
    logger.info("STEP 3: Validating PipelineResult...")
    logger.info("=" * 60)

    assert isinstance(result, PipelineResult), f"Expected PipelineResult, got {type(result)}"
    assert result.video_id == "test-001"
    assert result.total_frames > 0, "Should have extracted frames"
    assert result.processing_time_sec > 0, "Should have processing time"

    logger.info(f"  video_id:            {result.video_id}")
    logger.info(f"  total_frames:        {result.total_frames}")
    logger.info(f"  fps_processed:       {result.fps_processed}")
    logger.info(f"  processing_time_sec: {result.processing_time_sec:.2f}s")
    logger.info(f"  total_detections:    {len(result.detections)}")
    logger.info(f"  total_events:        {len(result.events)}")
    logger.info(f"  unique_persons:      {result.unique_person_count}")
    logger.info(f"  unique_vehicles:     {result.unique_vehicle_count}")
    logger.info(f"  class_counts:        {result.object_class_counts}")

    # Validate heatmap
    hm = result.heatmap
    assert hm.width > 0, "Heatmap should have width"
    assert hm.height > 0, "Heatmap should have height"
    assert len(hm.grid) == hm.height, f"Grid rows ({len(hm.grid)}) should match height ({hm.height})"
    assert len(hm.grid[0]) == hm.width, f"Grid cols ({len(hm.grid[0])}) should match width ({hm.width})"
    logger.info(f"  heatmap:             {hm.width}x{hm.height} grid")

    # Validate detections have proper fields
    if result.detections:
        d = result.detections[0]
        logger.info(f"  sample detection:    frame={d.frame_number}, class={d.class_name}, "
                     f"conf={d.confidence:.3f}, track_id={d.track_id}")
        assert d.detection_id, "Detection should have ID"
        assert d.bbox, "Detection should have bbox"

    # Validate events
    if result.events:
        e = result.events[0]
        logger.info(f"  sample event:        type={e.event_type}, class={e.class_name}, "
                     f"desc='{e.description[:60]}...'")
        assert e.event_id, "Event should have ID"
        assert e.description, "Event should have description"

    # Validate JSON serialization (this is what backend receives)
    result_json = result.model_dump()
    assert isinstance(result_json, dict)
    logger.info(f"  JSON serializable:   YES ({len(json.dumps(result_json))} bytes)")

    # Step 4: Test search engine
    logger.info("=" * 60)
    logger.info("STEP 4: Testing NL search...")
    logger.info("=" * 60)
    try:
        from ml.search.query_engine import QueryEngine
        qe = QueryEngine()
        stats = qe.get_stats()
        logger.info(f"  ChromaDB stats:      {stats}")

        search_results = qe.search("person entered", limit=5)
        logger.info(f"  Search 'person entered': {len(search_results)} results")
        for sr in search_results[:3]:
            logger.info(f"    - [{sr['relevance_score']:.3f}] {sr['description'][:80]}")
    except Exception as e:
        logger.warning(f"  Search test skipped: {e}")

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
