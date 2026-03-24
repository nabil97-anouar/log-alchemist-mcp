#!/usr/bin/env python3
"""Local smoke test for the Log Alchemist MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


ROOT = Path(__file__).parent
DEFAULT_SAMPLE_LOG = """\
[2026-03-24 10:41:12] slurmstepd: error: Detected 1 oom_kill event in StepId=2487.0 on cn30
[2026-03-24 10:41:12] srun: error: cn30: task 3: Exited with exit code 1
[2026-03-24 10:41:12] job 2487 rank 3 gpu 0 NCCL WARN NET/IB : Connection closed by remote peer
[2026-03-24 10:41:13] RuntimeError: CUDA out of memory
"""


def _load_log_text(log_file: str | None) -> str:
    if not log_file:
        return DEFAULT_SAMPLE_LOG
    return Path(log_file).read_text(encoding="utf-8")


async def _run_smoke_test(log_text: str, env_context: str, surface: str, exercise_sessions: bool) -> None:
    transport = StdioTransport(
        command=sys.executable,
        args=["mcp_server.py"],
        cwd=str(ROOT),
    )
    client = Client(transport)

    async with client:
        tools = await client.list_tools()
        prompts = await client.list_prompts()
        resources = await client.list_resources()

        print("MCP server reachable.")
        print(f"Tools ({len(tools)}): {', '.join(tool.name for tool in tools)}")
        print(f"Prompts ({len(prompts)}): {', '.join(prompt.name for prompt in prompts)}")
        print(f"Resources ({len(resources)}): {', '.join(str(resource.uri) for resource in resources)}")

        incident = await client.call_tool("extract_incident", {"log_text": log_text})
        scope = await client.call_tool(
            "assess_incident_scope",
            {"log_text": log_text, "env_context": env_context},
        )
        bundle = await client.call_tool(
            "build_incident_bundle",
            {"log_text": log_text, "env_context": env_context},
        )
        signatures = await client.call_tool("parse_hpc_failure_artifacts", {"log_text": log_text})
        manifest = await client.call_tool(
            "build_collection_manifest",
            {"log_text": log_text, "env_context": env_context},
        )
        gaps = await client.call_tool(
            "assess_evidence_gaps",
            {"log_text": log_text, "env_context": env_context},
        )
        companions = await client.call_tool("suggest_mcp_companions", {"log_text": log_text})
        plan = await client.call_tool(
            "build_agentic_response_plan",
            {"log_text": log_text, "env_context": env_context},
        )
        mission = await client.call_tool(
            "generate_playwright_mission",
            {"log_text": log_text, "env_context": env_context, "surface": surface},
        )
        session = None
        sessions = None
        if exercise_sessions:
            session = await client.call_tool(
                "open_incident_session",
                {"log_text": log_text, "env_context": env_context, "title": "Smoke test incident"},
            )
            sessions = await client.call_tool("list_incident_sessions", {"limit": 5})

    print("\nExtracted incident:")
    print(json.dumps(incident.data, indent=2))

    print("\nIncident scope:")
    print(json.dumps(scope.data, indent=2))

    print("\nIncident bundle summary:")
    print(
        json.dumps(
            {
                "tags": bundle.data.get("incident", {}).get("tags"),
                "topology": bundle.data.get("topology"),
                "runbook_preview": bundle.data.get("runbooks", "")[:500],
            },
            indent=2,
        )
    )

    print("\nEvidence gaps:")
    print(json.dumps(gaps.data, indent=2))

    print("\nStructured HPC signatures:")
    print(json.dumps(signatures.data, indent=2))

    print("\nCollection manifest:")
    print(
        json.dumps(
            {
                "ready_for_llm": manifest.data.get("ready_for_llm"),
                "grounding_score": manifest.data.get("grounding_score"),
                "confidence_label": manifest.data.get("confidence_label"),
                "filesystem": manifest.data.get("filesystem", [])[:2],
                "shell": manifest.data.get("shell", [])[:2],
                "playwright": manifest.data.get("playwright", [])[:2],
            },
            indent=2,
        )
    )

    print("\nCompanion MCP servers:")
    print(json.dumps(companions.data, indent=2))

    print("\nAgentic response plan:")
    print(json.dumps(plan.data, indent=2))

    print("\nPlaywright mission:")
    mission_text = mission.data if isinstance(mission.data, str) else str(mission.data)
    print(mission_text)

    if exercise_sessions and session and sessions:
        print("\nOpened incident session:")
        print(json.dumps(session.data, indent=2))

        print("\nRecent sessions:")
        print(json.dumps(sessions.data, indent=2))
    else:
        print("\nSessions:")
        print("Skipped by default. Re-run with --exercise-sessions to test session persistence APIs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Log Alchemist MCP server.")
    parser.add_argument(
        "--log-file",
        help="Optional path to a real log file to test instead of the built-in sample.",
    )
    parser.add_argument(
        "--env-context",
        default="cluster=training-a100; scheduler=slurm; workload=distributed-pytorch",
        help="Optional environment context passed to the MCP tools.",
    )
    parser.add_argument(
        "--surface",
        default="Grafana dashboard",
        help="Browser surface used for the Playwright mission.",
    )
    parser.add_argument(
        "--exercise-sessions",
        action="store_true",
        help="Also open a session and list recent sessions. Disabled by default so smoke tests do not spam local session history.",
    )
    args = parser.parse_args()

    log_text = _load_log_text(args.log_file)
    asyncio.run(_run_smoke_test(log_text, args.env_context, args.surface, args.exercise_sessions))


if __name__ == "__main__":
    main()
