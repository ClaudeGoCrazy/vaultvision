"""
Test ML pipeline → Backend integration.
Runs the pipeline and POSTs the PipelineResult to the backend's ingest endpoint.
Auto-discovers the backend port by checking common ports.
"""
import sys
import json
import uuid
import logging
import urllib.request
import urllib.error
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("test_backend_integration")


def find_backend():
    """Auto-discover backend URL by checking common ports."""
    for port in [8001, 8000, 8080]:
        url = f"http://localhost:{port}"
        try:
            req = urllib.request.Request(f"{url}/openapi.json", method="GET")
            resp = urllib.request.urlopen(req, timeout=2)
            spec = json.loads(resp.read())
            if "/api/v1/ingest/pipeline-result" in spec.get("paths", {}):
                logger.info(f"Backend found at {url}")
                return url
        except Exception:
            continue
    return None


def upload_video(backend_url: str, video_path: str) -> str | None:
    """Upload a video to the backend and return the video_id."""
    boundary = uuid.uuid4().hex
    with open(video_path, "rb") as f:
        video_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample_surveillance.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + video_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{backend_url}/api/v1/videos/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result.get("video_id")
    except Exception as e:
        logger.warning(f"Upload failed: {e}")
        return None


def test_ingest_to_backend():
    """Run pipeline and POST result to backend."""
    from ml.pipeline.main import process_video
    from shared.schemas import PipelineResult as PR

    video_path = str(project_root / "ml" / "data" / "uploads" / "sample_surveillance.mp4")

    # Find backend
    backend_url = find_backend()

    # Run pipeline
    logger.info("Running ML pipeline...")
    test_video_id = f"ml-test-{uuid.uuid4().hex[:8]}"

    # If backend is up, upload video first to get a valid video_id
    if backend_url:
        logger.info("Backend found — uploading video first...")
        uploaded_id = upload_video(backend_url, video_path)
        if uploaded_id:
            test_video_id = uploaded_id
            logger.info(f"Video uploaded with id: {test_video_id}")

    result = process_video(
        video_path=video_path,
        video_id=test_video_id,
        video_filename="sample_surveillance.mp4",
        cleanup_frames=True,
    )
    logger.info(f"Pipeline done: {len(result.detections)} detections, {len(result.events)} events")

    # Validate against shared schema
    result_json = result.model_dump(mode="json")
    validated = PR.model_validate(result_json)
    logger.info(f"Schema validation: PASSED (video_id={validated.video_id})")

    payload = json.dumps(result_json).encode("utf-8")
    logger.info(f"Payload size: {len(payload):,} bytes")

    if not backend_url:
        logger.info("Backend not detected — skipping API integration tests")
        logger.info("PipelineResult JSON is valid and ready for ingestion when backend starts")
        return

    # POST to backend ingest endpoint
    ingest_url = f"{backend_url}/api/v1/ingest/pipeline-result"
    logger.info(f"POSTing to {ingest_url}...")
    try:
        req = urllib.request.Request(
            ingest_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read().decode("utf-8"))
        logger.info(f"Ingest response ({resp.status}): {json.dumps(body)[:300]}")
    except urllib.error.HTTPError as e:
        logger.warning(f"Ingest HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:
        logger.warning(f"Ingest error: {e}")

    # Verify via API
    for endpoint, label in [
        (f"/api/v1/videos/{test_video_id}/status", "Status"),
        (f"/api/v1/videos/{test_video_id}/detections", "Detections"),
        (f"/api/v1/videos/{test_video_id}/heatmap", "Heatmap"),
    ]:
        try:
            resp = urllib.request.urlopen(f"{backend_url}{endpoint}", timeout=10)
            data = json.loads(resp.read())
            if isinstance(data, list):
                logger.info(f"  {label}: {len(data)} items")
            elif isinstance(data, dict):
                logger.info(f"  {label}: {json.dumps(data)[:200]}")
        except Exception as e:
            logger.warning(f"  {label} error: {e}")

    # Test NL search
    try:
        search_payload = json.dumps({"query": "person entered", "limit": 5}).encode("utf-8")
        req = urllib.request.Request(
            f"{backend_url}/api/v1/query",
            data=search_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        body = json.loads(resp.read())
        logger.info(f"  Search: {body.get('total_results', 0)} results for 'person entered'")
    except Exception as e:
        logger.warning(f"  Search error: {e}")

    logger.info("Integration test complete!")


if __name__ == "__main__":
    test_ingest_to_backend()
