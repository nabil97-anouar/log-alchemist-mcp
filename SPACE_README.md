---
title: Log Alchemist
emoji: 🧙
colorFrom: blue
colorTo: gray
sdk: gradio
python_version: "3.11"
app_file: log-alchemist-mcp/app.py
pinned: false
short_description: Agentic incident response for HPC logs with Gradio + MCP-aware workflows.
---

# Log Alchemist — Agentic Incident Response for HPC

Log Alchemist is an HPC-first incident triage app for noisy distributed failures.

This Space hosts the Gradio UI surface of the project:

- multi-log upload and OCR
- deterministic incident extraction
- evidence gaps and collection manifests
- agentic response planning
- optional LLM remediation layered on structured evidence

Notes:

- OCR requires `tesseract-ocr`, which is installed via `packages.txt`.
- The UI app entrypoint is `log-alchemist-mcp/app.py`.
- The dedicated FastMCP server remains part of the repo, but the main goal of this Space is to expose the Gradio app.
