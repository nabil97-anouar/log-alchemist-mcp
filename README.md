# Log Alchemist — Agentic Incident Response for HPC

Log Alchemist is an MCP-native, agentic incident response assistant for distributed training and infrastructure failures.
It is designed for the cases where pasting a raw log blob into a generic chat tool is not enough:
multi-rank failures, noisy repeated templates, sensitive internal data, recurring cluster-specific incidents, and investigations that require companion tools through MCP.

## Why This Project Is Different

- HPC-first, not generic observability chat
- MCP-native architecture with dedicated tools, resources, prompts, and stateful incident sessions
- Agentic investigation plans instead of one-shot summaries
- Multi-file bundle analysis for distributed failures across ranks and nodes
- Deterministic template mining to separate repetitive noise from likely root-cause lines
- Topology extraction for hosts, ranks, GPUs, interfaces, and job IDs
- Structured parsing of TorchElastic, Slurm, NCCL, and platform-orchestrator artifacts
- Scope assessment that distinguishes access/bootstrap traces, platform-orchestrator incidents, and true runtime failures before launching the wrong workflow
- Companion-server collection manifests for filesystem, shell, Playwright, docs, and telemetry MCPs
- Private local incident memory so the system improves on your own cluster’s failures over time
- Playwright-ready browser missions for Grafana, scheduler portals, Kibana, and cluster dashboards
- Optional LLM remediation layered on top of structured evidence, not used as the only analysis engine

## What It Does

- Redacts common secrets from logs before deeper processing
- Extracts grounded incident summaries with line-referenced top signals
- Detects whether the current sample is an access/bootstrap issue, a platform-orchestrator incident, a runtime HPC failure, or still too ambiguous
- Builds advanced incident bundles with:
  - incident summary
  - topology hints
  - template mining
  - structured HPC failure signatures
  - deterministic runbooks
  - official guidance links
  - local memory lookups
- Builds collection manifests that translate one incident into concrete filesystem, shell, docs, telemetry, and Playwright tasks
- Surfaces a grounding score so agents can tell when they still need more evidence before trusting an LLM-heavy answer
- Opens persistent incident sessions so MCP clients can revisit the same case without repasting the full logs
- Stores incidents locally and finds similar past failures
- Assesses evidence gaps before deeper reasoning
- Builds agentic response plans across MCP companions
- Generates MCP investigation prompts for companion toolchains
- Generates Playwright scouting missions for browser-only operational surfaces
- Exposes all of this through:
  - a Gradio UI
  - a dedicated FastMCP server

## MCP Architecture

This project now has two primary surfaces:

1. `log-alchemist-mcp/mcp_server.py`
   A dedicated FastMCP server with:
   - Tools
   - Resources
   - Prompts
   - Stateful incident sessions

2. `log-alchemist-mcp/app.py`
   A Gradio UI for manual triage, bundle generation, evidence gaps, agentic response plans, Playwright missions, memory operations, and optional LLM remediation.

The dedicated MCP server is the primary MCP story for this project.
The UI exists to make testing and operator workflows easier, but the server is where the project becomes more than a log summarizer:
it can maintain state across multiple artifacts and expose that state back to agents through MCP resources.

## File Map

Core product files:

- `log-alchemist-mcp/app.py`
  Gradio UI plus the deterministic incident-analysis engine used by both the UI and the MCP server
- `log-alchemist-mcp/mcp_server.py`
  The real FastMCP server that exposes tools, resources, prompts, and session-aware workflows
- `log-alchemist-mcp/incident_sessions.py`
  Local persistence and resource materialization for multi-artifact incident sessions
- `log-alchemist-mcp/smoke_test_mcp.py`
  The fastest way to validate the MCP server end-to-end without wiring up another MCP client first
- `log-alchemist-mcp/requirements.txt`
  Minimal runtime dependencies for the app and MCP server
- `log-alchemist-mcp/.env.example`
  Optional template for `HF_TOKEN` if you want the LLM remediation path

Agent and MCP client convenience files:

- `agent.json`
  Repo-root Tiny Agent config that points at `log-alchemist-mcp/mcp_server.py`
- `log-alchemist-mcp/agent.local.json`
  Tiny Agent config intended to be run from inside `log-alchemist-mcp/`
- `log-alchemist-mcp/agent.local.private.json`
  Local/private Tiny Agent example that also wires in Desktop Commander
- `PROMPT.md`
  Root agent instructions for repo-level agent runs
- `log-alchemist-mcp/PROMPT.md`
  Prompt/instructions for local incident-analysis agent workflows

Optional editor integration files:

- `.continue/mcpServers/log_alchemist.yaml`
  Example Continue MCP config for Log Alchemist
- `.continue/mcpServers/playwright-mcp.yaml`
  Example Continue MCP config for the Playwright companion server

Packaging / environment files:

- `pyproject.toml`
  Project metadata and dependency declaration
- `uv.lock`
  Lockfile for reproducible installs if you use `uv`
- `.gitignore`
  Prevents local memory, sessions, env files, and editor-specific files from being committed accidentally

## Requirements

- Python 3.11+
- `pip`
- Optional OCR support:
  - `pytesseract`
  - Tesseract binary installed on your machine
- Optional LLM remediation:
  - Hugging Face token in `log-alchemist-mcp/.env` as `HF_TOKEN=...`

## Setup

```bash
python3 -m venv log-alchemist-mcp/.venv
source log-alchemist-mcp/.venv/bin/activate
python -m pip install -r log-alchemist-mcp/requirements.txt
```

Create `log-alchemist-mcp/.env` from the example file:

```bash
HF_TOKEN=your_huggingface_token_here
```

## Run The UI

```bash
source log-alchemist-mcp/.venv/bin/activate
cd log-alchemist-mcp
python app.py
```

The UI lets you:

- upload one or many log files
- OCR screenshots of logs
- extract incidents
- generate incident cards
- build advanced incident bundles
- parse structured TorchElastic, Slurm, NCCL, and platform signatures
- build collection manifests for companion tools
- assess evidence gaps
- build agentic response plans
- generate Playwright missions
- save incidents to local memory
- find similar past incidents
- generate MCP investigation prompts for companion agents

## Run The Dedicated MCP Server

```bash
source log-alchemist-mcp/.venv/bin/activate
cd log-alchemist-mcp
python mcp_server.py
```

By default it runs over stdio.

That means the process is supposed to wait for an MCP client. It is healthy if it starts and then appears idle.
Do not treat a noisy Ctrl-C shutdown from a raw `stdio` server as the real health check. Use `smoke_test_mcp.py` for an actual validation path.

If you only care about the Gradio app, you can ignore this section entirely. The dedicated MCP server exists for MCP clients, stateful incident sessions, and tool/resource/prompt access outside the UI.

To smoke-test it locally:

```bash
source log-alchemist-mcp/.venv/bin/activate
cd log-alchemist-mcp
python smoke_test_mcp.py
```

That default smoke test checks deterministic extraction, manifests, evidence gaps, companion routing, and mission generation without polluting local incident-session history.

To test it against a real incident log:

```bash
source log-alchemist-mcp/.venv/bin/activate
cd log-alchemist-mcp
python smoke_test_mcp.py --log-file /absolute/path/to/incident.log
```

To also exercise the session APIs explicitly:

```bash
source log-alchemist-mcp/.venv/bin/activate
cd log-alchemist-mcp
python smoke_test_mcp.py --exercise-sessions
```

To run it as streamable HTTP:

```bash
source log-alchemist-mcp/.venv/bin/activate
cd log-alchemist-mcp
LOG_ALCHEMIST_MCP_TRANSPORT=streamable-http LOG_ALCHEMIST_MCP_PORT=8001 python mcp_server.py
```

## Exposed MCP Primitives

### Tools

- `extract_incident`
- `assess_incident_scope`
- `build_incident_bundle`
- `parse_hpc_failure_artifacts`
- `build_collection_manifest`
- `assess_evidence_gaps`
- `build_agentic_response_plan`
- `generate_playwright_mission`
- `open_incident_session`
- `append_incident_artifact`
- `list_incident_sessions`
- `get_incident_session`
- `resolve_incident_session`
- `suggest_mcp_companions`
- `save_incident_to_memory`
- `find_similar_incidents`
- `suggest_fix`

### Resources

- `playbook://tag/{tag}`
- `memory://recent`
- `guidance://official-hpc-docs`
- `agent://roles`
- `session://recent`
- `incident://session/{session_id}/summary`
- `incident://session/{session_id}/bundle`
- `incident://session/{session_id}/evidence-gaps`
- `incident://session/{session_id}/collection-manifest`
- `incident://session/{session_id}/artifacts`
- `incident://session/{session_id}/playwright/{surface}`

### Prompts

- `triage_hpc_incident`
- `escalation_handoff`
- `investigate_with_companion_servers`
- `playwright_incident_scout`
- `investigate_incident_session`
- `collect_incident_evidence`
- `agentic_incident_response`

## Why The MCP Server Matters

Without the server, Log Alchemist would mostly be a local UI and a set of parsing functions.
With the server, it becomes an incident-response control plane that other agents can use.

That added value comes from four things:

- stateful sessions, so an agent can open a case once and keep adding evidence
- resource URIs, so other agents can read the current bundle, evidence gaps, manifest, or attached artifacts without repasting logs
- collection manifests, so companion MCP servers receive concrete tasks instead of vague “go investigate” instructions
- HPC-specific parsing, so TorchElastic, Slurm, NCCL, and platform lifecycle signals are surfaced before the LLM improvises
- boundary checks, so the system can stop and say “this is SSH/bootstrap”, “this is a platform incident”, or “this is a real runtime failure” before routing the investigation

Sessions are most valuable when you append multiple artifacts over time.
For a one-shot smoke test or a tiny pasted log, they are optional and intentionally not exercised by default.

## Best MCP Pairing

The strongest real-world setup is:

- Log Alchemist MCP
  for structured HPC incident extraction, bundle generation, and memory
- Filesystem MCP
  for adjacent logs, configs, and saved diagnostics
- Shell MCP
  for deterministic next commands
- Playwright MCP
  for Grafana, Slurm web UIs, job portals, Kibana, storage panels, and browser-only evidence collection
- Browser/docs MCP
  for official Determined, NCCL, PyTorch Elastic, and Slurm references
- Optional telemetry/metrics MCP
  for GPU/fabric/scheduler correlation

This is the core product thesis:
Log Alchemist should be the agentic HPC incident brain inside a larger MCP toolchain.

## Tiny Agents

You can run a Tiny Agent directly against the dedicated MCP server with:

- `log-alchemist-mcp/agent.local.json`
- `log-alchemist-mcp/agent.local.private.json`

Run those from the `log-alchemist-mcp/` directory so `python mcp_server.py` resolves correctly.

## Continue

This repo includes:

- `.continue/mcpServers/log_alchemist.yaml`
- `.continue/mcpServers/playwright-mcp.yaml`

What this is for:

- If you use Continue as your in-editor agent, these files let Continue talk directly to Log Alchemist MCP and optional companion MCP servers.
- That means a Continue agent can call the same tools you tested with `smoke_test_mcp.py` instead of only chatting over pasted text.
- This is optional convenience, not the core product.

What is currently true:

- The core validated path in this repo is the Gradio UI plus `python smoke_test_mcp.py`.
- The Continue integration files are included as optional examples, but they are not required to use or validate Log Alchemist.
- The Continue path was not the primary validation path for this repo; treat it as convenience wiring rather than the core proof that the project works.
- If you do not use Continue, you can ignore the entire `.continue/` folder without losing any core functionality.

## Quick Validation

1. Activate the venv.
2. Run `python -m py_compile log-alchemist-mcp/app.py`.
3. Launch the UI and verify:
   - incident extraction works
   - bundle generation works
   - structured artifact parsing works
   - collection manifest generation works
   - evidence gap assessment works
   - agentic response plan generation works
   - Playwright mission generation is skipped when hosts/jobs/interfaces are too weak to support a browser workflow
   - local memory save/find works
   - final LLM remediation is blocked when `ready_for_llm` is false
   - MCP workflow prompt generation works
4. Run `cd log-alchemist-mcp && python smoke_test_mcp.py`.
5. Optionally run `cd log-alchemist-mcp && python smoke_test_mcp.py --exercise-sessions`.
6. Optionally, in a separate MCP client such as Tiny Agent or Continue, confirm that tools, resources, prompts, and session resource templates are discoverable.

## Current Product Direction

The current focus is to make Log Alchemist special in three ways:

- better than a generic LLM at noisy distributed incidents
- strongly MCP-native instead of MCP-as-a-demo
- useful for private, cluster-specific incident workflows
- able to coordinate multi-step investigations instead of only summarizing a single log blob

## Roadmap

- Structured parsers for TorchElastic and NCCL RAS artifacts
- Cluster-specific resource packs and runbooks
- Better recurrence ranking and local incident memory
- Automated tests for the deterministic core
- Cleaner separation between core analysis library, UI, and MCP server
