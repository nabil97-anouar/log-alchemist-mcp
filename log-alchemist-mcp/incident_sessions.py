#!/usr/bin/env python3
"""Stateful incident sessions for the Log Alchemist MCP server."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import app


SESSIONS_DIR = Path(__file__).parent / ".incident_sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_sessions_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _resource_hints(session_id: str) -> dict:
    return {
        "summary": f"incident://session/{session_id}/summary",
        "bundle": f"incident://session/{session_id}/bundle",
        "evidence_gaps": f"incident://session/{session_id}/evidence-gaps",
        "collection_manifest": f"incident://session/{session_id}/collection-manifest",
        "artifacts": f"incident://session/{session_id}/artifacts",
        "playwright_grafana": f"incident://session/{session_id}/playwright/Grafana%20dashboard",
    }


def _artifact_record(name: str, artifact_text: str, artifact_kind: str) -> dict:
    clean = app.redact_secrets(artifact_text)
    incident = app.extract_incident(clean)
    return {
        "artifact_id": uuid4().hex[:12],
        "name": name.strip() or "artifact",
        "kind": artifact_kind.strip() or "log",
        "added_at": _now_iso(),
        "line_count": incident["line_count"],
        "tags": incident["tags"],
        "top_signals": incident["top_signals"][:5],
        "text": clean,
    }


def _compose_session_text(artifacts: list[dict]) -> str:
    parts = []
    for artifact in artifacts:
        parts.append(
            "\n".join(
                [
                    f"===== ARTIFACT: {artifact['name']} ({artifact['kind']}) =====",
                    artifact["text"].strip(),
                ]
            ).strip()
        )
    return "\n\n".join(part for part in parts if part.strip())


def _materialize_session(
    *,
    session_id: str,
    title: str,
    env_context: str,
    status: str,
    artifacts: list[dict],
    created_at: str,
    updated_at: str | None = None,
    resolution_notes: str = "",
) -> dict:
    combined_text = _compose_session_text(artifacts)
    bundle = app.build_incident_bundle(combined_text, env_context)
    return {
        "session_id": session_id,
        "title": title or "Untitled HPC incident",
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at or _now_iso(),
        "env_context": env_context,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "combined_text": combined_text,
        "bundle": bundle,
        "collection_manifest": app.build_collection_manifest(combined_text, env_context),
        "companion_servers": app.suggest_mcp_companions(combined_text)["recommended_servers"],
        "evidence_gaps": app.assess_evidence_gaps(combined_text, env_context),
        "resolution_notes": resolution_notes,
        "resource_hints": _resource_hints(session_id),
    }


def _save_session(session: dict) -> None:
    _ensure_sessions_dir()
    _session_path(session["session_id"]).write_text(json.dumps(session, indent=2), encoding="utf-8")


def get_incident_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        return {
            "status": "error",
            "message": f"Incident session {session_id} was not found.",
            "session_id": session_id,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def open_incident_session(log_text: str, env_context: str = "", title: str = "") -> dict:
    if not log_text.strip():
        return {"status": "error", "message": "No logs provided."}

    session_id = f"sess-{uuid4().hex[:10]}"
    created_at = _now_iso()
    artifacts = [_artifact_record("initial logs", log_text, "log")]
    session = _materialize_session(
        session_id=session_id,
        title=title.strip() or "Initial HPC incident",
        env_context=env_context.strip(),
        status="open",
        artifacts=artifacts,
        created_at=created_at,
        updated_at=created_at,
    )
    _save_session(session)
    return {
        "status": "created",
        "message": f"Created incident session {session_id}.",
        "session_id": session_id,
        "title": session["title"],
        "incident": session["bundle"]["incident"],
        "artifact_count": session["artifact_count"],
        "resource_hints": session["resource_hints"],
    }


def append_incident_artifact(
    session_id: str,
    artifact_name: str,
    artifact_text: str,
    artifact_kind: str = "log",
) -> dict:
    session = get_incident_session(session_id)
    if session.get("status") == "error":
        return session
    if not artifact_text.strip():
        return {"status": "error", "message": "No artifact text provided.", "session_id": session_id}

    artifacts = list(session.get("artifacts", []))
    artifacts.append(_artifact_record(artifact_name, artifact_text, artifact_kind))
    updated = _materialize_session(
        session_id=session["session_id"],
        title=session.get("title", ""),
        env_context=session.get("env_context", ""),
        status="open",
        artifacts=artifacts,
        created_at=session.get("created_at", _now_iso()),
        updated_at=_now_iso(),
        resolution_notes=session.get("resolution_notes", ""),
    )
    _save_session(updated)
    return {
        "status": "updated",
        "message": f"Appended {artifact_name!r} to {session_id}.",
        "session_id": session_id,
        "artifact_count": updated["artifact_count"],
        "incident": updated["bundle"]["incident"],
        "resource_hints": updated["resource_hints"],
    }


def list_incident_sessions(limit: int = 10) -> dict:
    _ensure_sessions_dir()
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        sessions.append(
            {
                "session_id": session.get("session_id"),
                "title": session.get("title"),
                "status": session.get("status"),
                "updated_at": session.get("updated_at"),
                "artifact_count": session.get("artifact_count", 0),
                "tags": session.get("bundle", {}).get("incident", {}).get("tags", []),
                "top_signals": session.get("bundle", {}).get("incident", {}).get("top_signals", [])[:3],
            }
        )
    return {
        "status": "ok",
        "count": len(sessions),
        "sessions": sessions[:limit],
    }


def resolve_incident_session(session_id: str, resolution_notes: str, status: str = "mitigated") -> dict:
    session = get_incident_session(session_id)
    if session.get("status") == "error":
        return session

    session["status"] = status
    session["resolution_notes"] = resolution_notes.strip()
    session["updated_at"] = _now_iso()
    memory_result = app.save_incident_to_memory(session.get("combined_text", ""), session.get("env_context", ""))
    session["memory_result"] = memory_result
    _save_session(session)
    return {
        "status": "resolved",
        "message": f"Resolved incident session {session_id}.",
        "session_id": session_id,
        "memory_result": memory_result,
    }


def read_session_section(session_id: str, section: str) -> str:
    session = get_incident_session(session_id)
    if session.get("status") == "error":
        return json.dumps(session, indent=2)

    mapping = {
        "summary": {
            "session_id": session["session_id"],
            "title": session["title"],
            "status": session["status"],
            "artifact_count": session["artifact_count"],
            "incident": session["bundle"]["incident"],
            "resource_hints": session["resource_hints"],
        },
        "bundle": session["bundle"],
        "evidence-gaps": session["evidence_gaps"],
        "collection-manifest": session["collection_manifest"],
        "artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "name": artifact["name"],
                "kind": artifact["kind"],
                "added_at": artifact["added_at"],
                "line_count": artifact["line_count"],
                "tags": artifact["tags"],
                "top_signals": artifact["top_signals"],
            }
            for artifact in session["artifacts"]
        ],
    }
    return json.dumps(mapping[section], indent=2)


def session_playwright_mission(session_id: str, surface: str) -> str:
    session = get_incident_session(session_id)
    if session.get("status") == "error":
        return json.dumps(session, indent=2)
    return app.generate_playwright_mission(
        session.get("combined_text", ""),
        session.get("env_context", ""),
        surface,
    )
