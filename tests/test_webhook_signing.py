"""Tests for webhook payload signing.

Covers:
- sign_payload produces a valid HMAC-SHA256 hex digest
- Different secrets produce different signatures
- Different payloads produce different signatures
- Signature verification round-trip
"""

from __future__ import annotations

import hashlib
import hmac
import json

from libs.webhooks import sign_payload


def test_sign_payload_returns_hex_string():
    """sign_payload returns a non-empty hex string."""
    sig = sign_payload("secret", b"hello")
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA-256 hex digest is 64 chars


def test_sign_payload_deterministic():
    """Same secret + payload always produces the same signature."""
    sig1 = sign_payload("my-secret", b'{"event_type":"processing_run.completed"}')
    sig2 = sign_payload("my-secret", b'{"event_type":"processing_run.completed"}')
    assert sig1 == sig2


def test_sign_payload_different_secrets_differ():
    """Different secrets produce different signatures for the same payload."""
    payload = b'{"event_type":"processing_run.created"}'
    sig1 = sign_payload("secret-a", payload)
    sig2 = sign_payload("secret-b", payload)
    assert sig1 != sig2


def test_sign_payload_different_payloads_differ():
    """Different payloads produce different signatures for the same secret."""
    secret = "shared-secret"
    sig1 = sign_payload(secret, b'{"status":"completed"}')
    sig2 = sign_payload(secret, b'{"status":"failed"}')
    assert sig1 != sig2


def test_sign_payload_matches_standard_hmac():
    """sign_payload output matches stdlib hmac.new directly."""
    secret = "verification-secret"
    payload = b'{"event_type":"processing_run.running","status":"running"}'
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert sign_payload(secret, payload) == expected


def test_build_run_event_payload_fields():
    """build_run_event_payload returns all required fields."""
    from libs.events import RunEventType
    from libs.webhooks import build_run_event_payload

    payload = build_run_event_payload(
        RunEventType.completed,
        project_id=1,
        video_id=42,
        processing_run_id=7,
        status="completed",
        artifact_types=["state", "detections"],
    )

    assert payload["event_type"] == "processing_run.completed"
    assert payload["project_id"] == 1
    assert payload["video_id"] == 42
    assert payload["processing_run_id"] == 7
    assert payload["status"] == "completed"
    assert payload["artifact_types"] == ["state", "detections"]
    assert "event_id" in payload
    assert "occurred_at" in payload


def test_build_run_event_payload_no_artifact_types():
    """artifact_types is omitted from the payload when not provided."""
    from libs.events import RunEventType
    from libs.webhooks import build_run_event_payload

    payload = build_run_event_payload(
        RunEventType.running,
        project_id=1,
        video_id=1,
        processing_run_id=1,
        status="running",
    )
    assert "artifact_types" not in payload


def test_build_run_event_payload_error_field():
    """error is included in the payload when provided."""
    from libs.events import RunEventType
    from libs.webhooks import build_run_event_payload

    payload = build_run_event_payload(
        RunEventType.failed,
        project_id=1,
        video_id=1,
        processing_run_id=1,
        status="error",
        error="pipeline exploded",
    )
    assert payload["error"] == "pipeline exploded"


def test_build_run_event_payload_unique_event_ids():
    """Each call to build_run_event_payload generates a unique event_id."""
    from libs.events import RunEventType
    from libs.webhooks import build_run_event_payload

    ids = {
        build_run_event_payload(
            RunEventType.created,
            project_id=1,
            video_id=1,
            processing_run_id=i,
            status="pending",
        )["event_id"]
        for i in range(5)
    }
    assert len(ids) == 5


def test_signature_verification_example():
    """Demonstrate the complete sign-then-verify pattern for documentation."""
    secret = "my-webhook-secret"
    payload_dict = {
        "event_type": "processing_run.completed",
        "video_id": 1,
        "processing_run_id": 3,
        "status": "completed",
    }
    # Sender signs with sorted keys for determinism.
    payload_bytes = json.dumps(payload_dict, sort_keys=True).encode()
    sent_signature = sign_payload(secret, payload_bytes)

    # Receiver recomputes the signature over the raw body.
    expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(sent_signature, expected)
