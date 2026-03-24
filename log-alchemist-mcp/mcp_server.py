#!/usr/bin/env python3
"""FastMCP server for Log Alchemist."""

import json
import os

from fastmcp import FastMCP

import app
import incident_sessions


mcp = FastMCP(
    "log-alchemist",
    instructions=(
        "Log Alchemist is an HPC incident response MCP server. "
        "Use its tools for deterministic incident extraction, bundle generation, "
        "local recurrence detection, and HPC-specific remediation planning."
    ),
)


@mcp.tool(description="Extract a grounded incident summary from HPC or infrastructure logs.")
def extract_incident(log_text: str) -> dict:
    return app.extract_incident(log_text)


@mcp.tool(description="Assess whether the current logs really look like an HPC runtime incident or a different class of problem such as access/bootstrap failure.")
def assess_incident_scope(log_text: str, env_context: str = "") -> dict:
    return app.assess_incident_scope(log_text, env_context)


@mcp.tool(description="Build an advanced incident bundle with topology, template mining, runbooks, and local memory hints.")
def build_incident_bundle(log_text: str, env_context: str = "") -> dict:
    return app.build_incident_bundle(log_text, env_context)


@mcp.tool(description="Parse TorchElastic, Slurm, NCCL, and platform-orchestrator failure artifacts out of raw incident logs.")
def parse_hpc_failure_artifacts(log_text: str) -> dict:
    return app.parse_hpc_failure_artifacts(log_text)


@mcp.tool(description="Build a concrete collection manifest for filesystem, shell, docs, telemetry, and Playwright companion MCP servers.")
def build_collection_manifest(log_text: str, env_context: str = "") -> dict:
    return app.build_collection_manifest(log_text, env_context)


@mcp.tool(description="Suggest which companion MCP servers would add the most value for this incident.")
def suggest_mcp_companions(log_text: str) -> dict:
    return app.suggest_mcp_companions(log_text)


@mcp.tool(description="Identify missing evidence and the highest-priority artifacts to collect before deeper reasoning.")
def assess_evidence_gaps(log_text: str, env_context: str = "") -> dict:
    return app.assess_evidence_gaps(log_text, env_context)


@mcp.tool(description="Build an agentic multi-stage response plan for the incident across MCP companions.")
def build_agentic_response_plan(log_text: str, env_context: str = "") -> dict:
    return app.build_agentic_response_plan(log_text, env_context)


@mcp.tool(description="Generate a Playwright mission for dashboard or portal inspection around the incident.")
def generate_playwright_mission(
    log_text: str,
    env_context: str = "",
    surface: str = "Grafana dashboard",
) -> str:
    return app.generate_playwright_mission(log_text, env_context, surface)


@mcp.tool(description="Create a stateful incident investigation session that an MCP client can revisit through session resources.")
def open_incident_session(log_text: str, env_context: str = "", title: str = "") -> dict:
    return incident_sessions.open_incident_session(log_text, env_context, title)


@mcp.tool(description="Append a new artifact, such as a sibling log, scheduler output, or error.json, to an existing incident session.")
def append_incident_artifact(
    session_id: str,
    artifact_name: str,
    artifact_text: str,
    artifact_kind: str = "log",
) -> dict:
    return incident_sessions.append_incident_artifact(session_id, artifact_name, artifact_text, artifact_kind)


@mcp.tool(description="List recent stateful incident sessions managed by Log Alchemist MCP.")
def list_incident_sessions(limit: int = 10) -> dict:
    return incident_sessions.list_incident_sessions(limit)


@mcp.tool(description="Read the full JSON payload for a previously opened incident session.")
def get_incident_session(session_id: str) -> dict:
    return incident_sessions.get_incident_session(session_id)


@mcp.tool(description="Mark an incident session resolved and persist its final state into local incident memory.")
def resolve_incident_session(session_id: str, resolution_notes: str, status: str = "mitigated") -> dict:
    return incident_sessions.resolve_incident_session(session_id, resolution_notes, status)


@mcp.tool(description="Store an incident in local private memory for future recurrence detection.")
def save_incident_to_memory(log_text: str, env_context: str = "") -> dict:
    return app.save_incident_to_memory(log_text, env_context)


@mcp.tool(description="Find similar previously saved incidents from local private memory.")
def find_similar_incidents(log_text: str, limit: int = 5) -> dict:
    return app.find_similar_incidents(log_text, limit=limit)


@mcp.tool(description="Generate an HPC-specific remediation plan using the configured Hugging Face model.")
def suggest_fix(log_text: str, env_context: str = "", model: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
    return app.suggest_fix(log_text, env_context, model)


@mcp.resource(
    "playbook://tag/{tag}",
    description="Read-only diagnostic playbook for a detected subsystem tag such as NCCL, GPU, InfiniBand, Scheduler, or Network.",
)
def playbook_for_tag(tag: str) -> str:
    normalized = tag.strip()
    return app.build_runbooks([normalized]) or app.build_runbooks(["Unclassified"])


@mcp.resource(
    "memory://recent",
    description="Recent locally stored incidents from Log Alchemist's private incident memory.",
    mime_type="application/json",
)
def recent_incidents() -> str:
    recent = app._read_incident_memory()[-10:]
    return json.dumps(recent, indent=2)


@mcp.resource(
    "guidance://official-hpc-docs",
    description="Curated official documentation URLs for Determined, NCCL, PyTorch Elastic, and Slurm troubleshooting.",
    mime_type="application/json",
)
def official_hpc_docs() -> str:
    flattened = []
    for tag, items in app.OFFICIAL_HPC_GUIDANCE.items():
        flattened.append({"tag": tag, "references": items})
    return json.dumps(flattened, indent=2)


@mcp.resource(
    "agent://roles",
    description="Agent roles for Log Alchemist's agentic incident-response workflow.",
    mime_type="application/json",
)
def agent_roles() -> str:
    roles = [
        {
            "agent": "Log Analyst Agent",
            "responsibility": "Ground the investigation using Log Alchemist tools and maintain evidence integrity.",
        },
        {
            "agent": "Evidence Collector Agent",
            "responsibility": "Use filesystem and shell MCP to gather missing artifacts and run deterministic commands.",
        },
        {
            "agent": "Playwright Scout Agent",
            "responsibility": "Use Playwright MCP to inspect dashboards, portals, and browser-only evidence surfaces.",
        },
        {
            "agent": "Incident Commander Agent",
            "responsibility": "Synthesize findings, prepare handoffs, and coordinate the final operator-facing response.",
        },
    ]
    return json.dumps(roles, indent=2)


@mcp.resource(
    "session://recent",
    description="Recent stateful incident sessions managed by Log Alchemist MCP.",
    mime_type="application/json",
)
def recent_sessions() -> str:
    return json.dumps(incident_sessions.list_incident_sessions(limit=10), indent=2)


@mcp.resource(
    "incident://session/{session_id}/summary",
    description="Summary view for a stateful incident session, including incident id, tags, and resource hints.",
    mime_type="application/json",
)
def incident_session_summary(session_id: str) -> str:
    return incident_sessions.read_session_section(session_id, "summary")


@mcp.resource(
    "incident://session/{session_id}/bundle",
    description="Full advanced incident bundle for a stateful incident session.",
    mime_type="application/json",
)
def incident_session_bundle(session_id: str) -> str:
    return incident_sessions.read_session_section(session_id, "bundle")


@mcp.resource(
    "incident://session/{session_id}/evidence-gaps",
    description="Evidence gaps and readiness assessment for a stateful incident session.",
    mime_type="application/json",
)
def incident_session_evidence_gaps(session_id: str) -> str:
    return incident_sessions.read_session_section(session_id, "evidence-gaps")


@mcp.resource(
    "incident://session/{session_id}/collection-manifest",
    description="Companion-server collection manifest for a stateful incident session.",
    mime_type="application/json",
)
def incident_session_collection_manifest(session_id: str) -> str:
    return incident_sessions.read_session_section(session_id, "collection-manifest")


@mcp.resource(
    "incident://session/{session_id}/artifacts",
    description="Artifacts currently attached to a stateful incident session.",
    mime_type="application/json",
)
def incident_session_artifacts(session_id: str) -> str:
    return incident_sessions.read_session_section(session_id, "artifacts")


@mcp.resource(
    "incident://session/{session_id}/playwright/{surface}",
    description="Session-specific Playwright mission for a chosen operational surface such as Grafana or a scheduler portal.",
)
def incident_session_playwright(session_id: str, surface: str) -> str:
    normalized_surface = surface.replace("%20", " ")
    return incident_sessions.session_playwright_mission(session_id, normalized_surface)


@mcp.prompt(description="Guide an MCP client through a deep HPC incident triage workflow using Log Alchemist tools and resources.")
def triage_hpc_incident(objective: str = "root cause deep dive") -> str:
    return (
        f"Use Log Alchemist to perform a {objective}.\n"
        "1. Call build_incident_bundle on the provided logs and environment context.\n"
        "2. Inspect scope_assessment first so you know whether this is access, platform/orchestrator, or runtime triage.\n"
        "3. Inspect incident tags, topology, template_mining, structured_signatures, official_guidance, and local_memory.\n"
        "4. Call build_collection_manifest to turn the incident into explicit filesystem, shell, docs, telemetry, and Playwright tasks.\n"
        "5. Read playbook://tag/{tag} resources for all detected tags.\n"
        "6. If the failure appears recurrent, call find_similar_incidents.\n"
        "7. Only after evidence is assembled, call suggest_fix if an LLM answer is needed.\n"
        "8. Cite line references from Log Alchemist outputs and distinguish evidence from inference."
    )


@mcp.prompt(description="Generate an escalation-ready handoff workflow for an HPC incident.")
def escalation_handoff(audience: str = "sre-oncall") -> str:
    return (
        f"Prepare a handoff for {audience}.\n"
        "First call build_incident_bundle, then summarize:\n"
        "- incident id and timestamps\n"
        "- affected hosts, ranks, GPUs, interfaces, and job ids\n"
        "- top signals with line references\n"
        "- suspected root causes and confidence\n"
        "- immediate safe actions\n"
        "- exact next diagnostics and owners\n"
        "Keep the handoff concise, operational, and evidence-based."
    )


@mcp.prompt(description="Explain how to pair Log Alchemist with filesystem, shell, browser/docs, or telemetry MCP servers.")
def investigate_with_companion_servers(objective: str = "collect adjacent evidence") -> str:
    return (
        f"Use Log Alchemist together with companion MCP servers to {objective}.\n"
        "Recommended order:\n"
        "1. Log Alchemist build_incident_bundle for structured incident context and scope assessment.\n"
        "2. Filesystem MCP to open neighboring logs, scheduler outputs, configs, startup hooks, and saved diagnostics.\n"
        "3. Shell MCP to run the deterministic next commands from the runbooks or collection manifest.\n"
        "4. Browser/docs MCP to confirm official Determined, NCCL, PyTorch Elastic, or Slurm guidance.\n"
        "5. Telemetry MCP, if available, to correlate the failure window with GPU/fabric or platform metrics.\n"
        "Always keep the final answer grounded in the evidence from Log Alchemist outputs."
    )


@mcp.prompt(description="Generate a browser-evidence collection workflow for Playwright MCP in an HPC incident.")
def playwright_incident_scout(surface: str = "Grafana dashboard") -> str:
    return (
        f"Use Playwright MCP to inspect the {surface} for an HPC incident.\n"
        "1. Open the relevant dashboard or portal.\n"
        "2. Set the time range around the failing timestamps.\n"
        "3. Filter to impacted hosts, jobs, GPUs, or interfaces.\n"
        "4. Capture screenshots and extract visible statuses, counters, or error markers.\n"
        "5. Summarize what the UI evidence confirms, contradicts, or leaves unresolved.\n"
        "6. Feed the results back into Log Alchemist before final remediation advice."
    )


@mcp.prompt(description="Run a stateful session-based investigation using Log Alchemist resources instead of repeatedly pasting the same logs.")
def investigate_incident_session(session_id: str, objective: str = "close the incident safely") -> str:
    return (
        f"Investigate session {session_id} to {objective}.\n"
        f"1. Read incident://session/{session_id}/summary.\n"
        f"2. Read incident://session/{session_id}/bundle and incident://session/{session_id}/collection-manifest.\n"
        f"3. If evidence is missing, append new artifacts with append_incident_artifact and reread the session resources.\n"
        f"4. Use the collection manifest to dispatch work to filesystem, shell, telemetry, and Playwright MCP servers.\n"
        f"5. When the evidence is strong enough, call suggest_fix or prepare a handoff.\n"
        f"6. After mitigation, call resolve_incident_session so the resolved case is remembered locally."
    )


@mcp.prompt(description="Translate a stateful incident session into companion-agent tasks for filesystem, shell, telemetry, and Playwright MCP servers.")
def collect_incident_evidence(session_id: str) -> str:
    return (
        f"Use incident://session/{session_id}/collection-manifest as the source of truth.\n"
        "Execute high-priority filesystem and shell tasks first.\n"
        "Then gather telemetry correlations and browser evidence from the listed Playwright surfaces.\n"
        "Append each newly collected artifact back into the session so Log Alchemist can recompute the evidence state."
    )


@mcp.prompt(description="Run Log Alchemist as an agentic incident-response loop instead of a single summarization step.")
def agentic_incident_response(objective: str = "stabilize the incident") -> str:
    return (
        f"Operate Log Alchemist as an agentic incident-response system to {objective}.\n"
        "Phase 1: build_incident_bundle and identify evidence-backed hypotheses.\n"
        "Phase 2: assess_evidence_gaps and collect missing artifacts.\n"
        "Phase 3: use build_collection_manifest and suggest_mcp_companions to route work to filesystem, shell, Playwright, docs, or telemetry MCP.\n"
        "Phase 4: generate_playwright_mission if browser-visible surfaces matter, or open_incident_session if the investigation will span multiple artifacts.\n"
        "Phase 5: call suggest_fix only after the evidence set is strong enough.\n"
        "Phase 6: save the incident to memory for future recurrence detection."
    )


if __name__ == "__main__":
    transport = os.getenv("LOG_ALCHEMIST_MCP_TRANSPORT", "stdio").strip().lower()
    port = int(os.getenv("LOG_ALCHEMIST_MCP_PORT", "8001"))

    if transport == "streamable-http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port, show_banner=False)
    elif transport == "sse":
        mcp.run(transport="sse", host="0.0.0.0", port=port, show_banner=False)
    else:
        mcp.run(show_banner=False)
