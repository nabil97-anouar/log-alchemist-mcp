import os
import json
import re
import hashlib
import html
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from dotenv import load_dotenv

load_dotenv()

# Optional OCR deps (works only if you have both pytesseract + tesseract installed)
try:
    import pytesseract
except Exception:
    pytesseract = None


# ============================
# Patterns / Heuristics
# ============================

_SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(bearer)\s+([a-z0-9\-_\.=]{15,})\b"), r"\1 [REDACTED_TOKEN]"),
    (re.compile(r"(?i)\b(hf_[a-z0-9]{20,})\b"), r"[REDACTED_HF_TOKEN]"),
    (re.compile(r"(?i)\b(sk-[a-z0-9]{20,})\b"), r"[REDACTED_KEY]"),
    (re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), r"[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), r"[REDACTED_IP]"),
]

# Broader HPC-ish tagger (generic, not only Slurm/DDP)
_ERROR_HINTS = [
    ("SSH", re.compile(r"(?i)\bssh\b|openssh|ProxyJump|known_hosts|authorized_keys|ssh_config")),
    ("GPU", re.compile(r"(?i)\bcuda\b|cudnn|cublas|xid|nvidia|nvlink|nvswitch|\bdetected\s+\d+\s+gpus?\b|visible GPUs|GPU-[0-9a-f-]{8,}")),
    ("OOM", re.compile(r"(?i)\bout of memory\b|\boom\b|oom-kill|killed process")),
    ("NCCL", re.compile(r"(?i)\bnccl\b|collective|allreduce|dist\.barrier")),
    ("TorchElastic", re.compile(r"(?i)\btorchelastic\b|ChildFailedError|ProcessFailure\(|TORCHELASTIC_ERROR_FILE|torch\.distributed\.elastic")),
    ("MPI", re.compile(r"(?i)\bmpirun\b|\bmpiexec\b|openmpi|mpich")),
    ("Scheduler", re.compile(r"(?i)\bslurm\b|srun|sbatch|squeue|slurmd|pbs|lsf|sge")),
    ("Platform", re.compile(r"(?i)\bdetermined\b|determinedai|allocation id|resource pool|workspace id|task_type_|agent_id=|slots limit|slots needed|running task container")),
    ("Container", re.compile(r"(?i)\bdocker\b|\bpodman\b|\bapptainer\b|\bsingularity\b|\bk8s\b|\bkubernetes\b|task container|container failed with non-zero exit code|resources failed with non-zero exit code")),
    ("Network", re.compile(r"(?i)\btimeout\b|\bconnection\b|\bdns\b|\brefused\b|unreachable|reset by peer")),
    ("InfiniBand", re.compile(r"(?i)\bibstat\b|ibv_|infiniband|mlx5|link_up|lid|sm_lid")),
    ("Disk/FS", re.compile(r"(?i)\bnfs\b|stale file handle|no space|quota|i/o error|read-only file system|vast")),
    ("Permission", re.compile(r"(?i)\bpermission denied\b|\beacces\b|operation not permitted")),
    ("Kernel/Driver", re.compile(r"(?i)\bdmesg\b|segfault|kernel panic|tainted|nvidia-peermem|mlx5_core|nouveau|kernel module")),
]
_HPC_SCOPE_HINTS = [
    ("GPU", re.compile(r"(?i)\bcuda\b|cudnn|cublas|xid|nvidia-smi|nvlink|nvswitch"), 0.2),
    ("NCCL", re.compile(r"(?i)\bnccl\b|allreduce|all_gather|reduce_scatter|dist\.barrier"), 0.22),
    ("TorchElastic", re.compile(r"(?i)\btorchelastic\b|ChildFailedError|ProcessFailure\(|TORCHELASTIC_ERROR_FILE"), 0.22),
    ("Scheduler", re.compile(r"(?i)\bslurm\b|slurmstepd|srun|sbatch|sacct|scontrol|JobID=|ExitCode=|DerivedExitCode="), 0.2),
    ("MPI", re.compile(r"(?i)\bmpirun\b|\bmpiexec\b|openmpi|mpich"), 0.18),
    ("InfiniBand", re.compile(r"(?i)\bibstat\b|ibv_devinfo|mlx5|rdma|infiniband|net/ib"), 0.2),
    ("Disk/FS", re.compile(r"(?i)\bnfs\b|stale file handle|lustre|beegfs|vast|quota"), 0.14),
]
_HPC_PLATFORM_HINTS = [
    ("Orchestrator", re.compile(r"(?i)\bdetermined\b|determinedai|allocation id|resource pool|task_type_|workspace id|agent_id=|slots needed|slots limit|running task container"), 0.28),
    ("GPU Allocation", re.compile(r"(?i)\bdetected\s+\d+\s+gpus?\b|visible GPUs|GPU-[0-9a-f-]{8,}"), 0.18),
    ("Container Lifecycle", re.compile(r"(?i)\bcopying files to container\b|task container|container failed with non-zero exit code|resources failed with non-zero exit code|image already found, skipping pull phase"), 0.18),
    ("Interactive Service", re.compile(r"(?i)\bServer listening on\b|Service of .* is available|Accepted publickey for "), 0.14),
]
_NON_HPC_SCOPE_HINTS = [
    ("SSH/OpenSSH", re.compile(r"(?i)\bssh\b|openssh|ProxyJump|known_hosts|authorized_keys|ssh_config"), 0.22),
    ("Authentication", re.compile(r"(?i)\bpermission denied\b|authentications that can continue|publickey|password:"), 0.24),
    ("SSH handshake", re.compile(r"(?i)\bKEXINIT\b|SSH2_MSG_|host key|channel 0:|stdio-forward"), 0.18),
]
_STRONG_SIGNAL_HINTS = [
    (re.compile(r"(?i)\bpermission denied\b"), 8),
    (re.compile(r"(?i)\bno more authentication methods to try\b"), 9),
    (re.compile(r"(?i)\bconnection refused\b|\bconnection reset\b|reset by peer|unreachable"), 8),
    (re.compile(r"(?i)\btimeout\b|\btimed out\b"), 7),
    (re.compile(r"(?i)\bchildfailederror\b|ProcessFailure\(|TORCHELASTIC_ERROR_FILE"), 9),
    (re.compile(r"(?i)\btraceback\b|\bruntimeerror\b|\bexception\b|\bfatal\b|\bsegfault\b"), 8),
    (re.compile(r"(?i)\bmodulenotfounderror\b|\bimporterror\b"), 9),
    (re.compile(r"(?i)\bno such file or directory\b"), 8),
    (re.compile(r"(?i)\bexited with exit code\b|\bnon-zero exit code\b"), 8),
    (re.compile(r"(?i)\bout of memory\b|oom-kill|killed process"), 8),
    (re.compile(r"(?i)\bnccl\b.*\bwarn\b|\bnccl\b.*\berror\b|\bnet/ib\b.*\bwarn\b|\bnet/ib\b.*\berror\b"), 8),
    (re.compile(r"(?i)\bslurmstepd: error\b|\bDerivedExitCode\b|\bExitCode\b"), 8),
    (re.compile(r"(?i)\bforcibly killing allocation\b|user requested kill|resources failed with non-zero exit code|terminated: allocation killed"), 8),
    (re.compile(r"(?i)\bAttempt to write login records by non-root user\b|Couldn'?t create pid file"), 7),
    (re.compile(r"(?i)\bService of .* is available\b|Accepted publickey for "), 4),
    (re.compile(r"(?i)\bexit code 137\b"), 7),
    (re.compile(r"(?i)\bkilled by signal\b"), 6),
]
_LOW_VALUE_SIGNAL_HINTS = [
    re.compile(r"(?i)^=+\s*FILE:"),
    re.compile(r"^\[[^@\]]+@[A-Za-z0-9._-]+[^\]]*\][#$%]\s"),
    re.compile(r"(?i)^cat\s+\S+"),
    re.compile(r"(?i)\breading configuration data\b"),
    re.compile(r"(?i)\bidentity file\b"),
    re.compile(r"(?i)\bexpanded UserKnownHostsFile\b"),
    re.compile(r"(?i)\brecord_hostkey\b|\bload_hostkeys\b"),
    re.compile(r"(?i)\bKEX algorithms\b|\bMACs ctos\b|\bMACs stoc\b|\bciphers ctos\b|\bciphers stoc\b"),
    re.compile(r"(?i)\blanguages ctos\b|\blanguages stoc\b|\bcompat_banner\b"),
    re.compile(r"(?i)\bnew stdio-forward\b"),
    re.compile(r"(?i)\bcopying files to container\b"),
    re.compile(r"(?i)\bimage already found, skipping pull phase\b"),
    re.compile(r"(?i)\bNCCL INFO\b"),
    re.compile(r"(?i)\bChannel \d+/\d+\b|\bRing \d+\b|\bTree \d+\b"),
    re.compile(r"(?i)\bproxyProgressAsync\b|\bsendConnect\b|\brecvConnect\b|\bConnected all rings\b"),
    re.compile(r"(?i)\bNET/IB: NCCL Dev\b|\bIb Alloc Size\b|\bCuda Host Alloc Size\b"),
    re.compile(r"(?i)\bGPU Direct RDMA Disabled\b"),
]

MAX_PREVIEW_LINES = 120
MAX_SIGNAL_LINES = 15
HEAD_PREVIEW_LINES = 40
TAIL_PREVIEW_LINES = 20
TOP_TEMPLATE_COUNT = 8
MIN_MEMORY_SIMILARITY = 0.2
INCIDENT_MEMORY_PATH = Path(__file__).parent / ".incident_memory.jsonl"
PLAYWRIGHT_SURFACES = [
    "Determined or platform UI",
    "Grafana dashboard",
    "Slurm or scheduler web UI",
    "Cluster job portal",
    "Kibana or log search UI",
    "Storage or fabric dashboard",
]
OFFICIAL_HPC_GUIDANCE = {
    "NCCL": [
        {
            "title": "NVIDIA NCCL environment variables",
            "url": "https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2162/user-guide/docs/env.html",
        },
        {
            "title": "NVIDIA NCCL RAS troubleshooting",
            "url": "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html",
        },
    ],
    "GPU": [
        {
            "title": "PyTorch torchrun elastic error summaries",
            "url": "https://docs.pytorch.org/docs/2.9/elastic/run.html",
        },
        {
            "title": "PyTorch Elastic error propagation",
            "url": "https://docs.pytorch.org/docs/2.9/elastic/errors.html",
        },
    ],
    "TorchElastic": [
        {
            "title": "PyTorch Elastic error propagation",
            "url": "https://docs.pytorch.org/docs/2.9/elastic/errors.html",
        },
        {
            "title": "PyTorch torchrun elastic error summaries",
            "url": "https://docs.pytorch.org/docs/2.9/elastic/run.html",
        },
    ],
    "Scheduler": [
        {
            "title": "Slurm troubleshooting guide",
            "url": "https://slurm.schedmd.com/troubleshoot.html",
        },
        {
            "title": "Slurm job exit codes",
            "url": "https://slurm.schedmd.com/job_exit_code.html",
        },
    ],
    "InfiniBand": [
        {
            "title": "NVIDIA NCCL environment variables",
            "url": "https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2162/user-guide/docs/env.html",
        },
    ],
    "Platform": [
        {
            "title": "Determined AI commands and shells",
            "url": "https://docs.determined.ai/0.17.3/features/commands-and-shells.html",
        },
        {
            "title": "Determined AI master configuration reference (resource pools)",
            "url": "https://docs.determined.ai/0.26.0/reference/deploy/config/master-config-reference.html",
        },
    ],
}
PRIMARY_SOURCE_LINKS = {
    "nccl_plugin_fallback": "https://docs.nvidia.com/networking/display/hpcxv224/nccl-rdma-sharp%2Bplugins",
    "torch_distributed_barrier": "https://docs.pytorch.org/docs/stable/distributed.html?highlight=init_process_group",
    "torch_ddp_device_mapping": "https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html",
}

_TIMESTAMP_RX = re.compile(r"\b(20\d{2}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2})\b")
_SOURCE_HEADER_RX = re.compile(r"^=+\s*FILE:\s*(.+?)\s*=+$")
_HOST_PATTERNS = [
    re.compile(r"(?i)\b(?:host|hostname|node)\s*[=:]\s*([A-Za-z0-9._-]+)\b"),
    re.compile(r"(?i)\bhost\s+([A-Za-z0-9._-]+)\b"),
    re.compile(r"(?i)\bsrun:\s+error:\s+([A-Za-z0-9._-]+):\s+task\b"),
    re.compile(r"^([A-Za-z][A-Za-z0-9._-]+):\d+:\d+\s"),
    re.compile(r"^\[[^@\]]+@([A-Za-z0-9._-]+)[^\]]*\][#$%]\s"),
]
_HOST_STOPWORDS = {
    "key", "keys", "algorithm", "algorithms", "name", "names", "file", "files", "port",
    "forward", "forwarding", "unknown", "client", "server", "hostkeyalgorithms",
}
_RANK_PATTERNS = [
    re.compile(r"(?i)\brank\s*[=:]?\s*(\d+)\b"),
    re.compile(r"(?i)\blocal_rank\s*[=:]?\s*(\d+)\b"),
]
_GPU_PATTERNS = [
    re.compile(r"(?i)\bgpu\s*[=:]?\s*(\d+)\b"),
    re.compile(r"(?i)\bcuda device\s*[=:]?\s*(\d+)\b"),
    re.compile(r"(?i)\b(GPU-[0-9a-fA-F-]{8,})\b"),
]
_JOB_PATTERNS = [
    re.compile(r"(?i)\b(?:job|jobid|slurm_job_id)\b\s*[=:]\s*([A-Za-z0-9._-]+)\b"),
    re.compile(r"(?i)\b(?:job|jobid|slurm_job_id)\b\s+([A-Za-z0-9._-]+)\b"),
    re.compile(r"(?i)/job(\d+)(?:/|\b)"),
    re.compile(r"(?i)\bjob(\d{3,})\b"),
]
_INTERFACE_PATTERN = re.compile(r"\b(?:bond\d+|ib\d+|mlx5_\d+|eth\d+|eno\d+|enp[a-z0-9]+s?\d*)\b")
_SIGNAL_PATTERN = re.compile(r"(?i)\berror\b|\bfailed\b|\bexception\b|\btraceback\b|\bwarn\b|\bfatal\b|modulenotfounderror|importerror|no such file or directory")
_SHELL_PROMPT_PATTERNS = [
    re.compile(r"^\[[^@\]]+@[A-Za-z0-9._-]+[^\]]*\][#$%]\s"),
    re.compile(r"^(?:\([^)]*\)\s*)?[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+(?::[^\s]*)?[#$%]\s"),
]
_CAT_COMMAND_RX = re.compile(r"(?:\]\$|[$#%])\s+cat\s+([^\s]+)")
_MEMORY_STOPWORDS = {
    "line", "with", "from", "this", "that", "have", "into", "your", "rank", "error", "failed",
    "warning", "warn", "fatal", "traceback", "runtime", "exception", "timeout", "connection",
    "process", "node", "host", "file", "logs", "cuda", "nccl", "gpu",
}
_NCCL_ENV_RX = re.compile(r"\b(NCCL_[A-Z0-9_]+)\s*[=:]\s*([^\s,;]+)")
_TORCHELASTIC_ERROR_FILE_RX = re.compile(r"(?i)\bTORCHELASTIC_ERROR_FILE\b(?:\s*[=:]\s*|\s+)([^\s,;]+)")
_TORCHELASTIC_MESSAGE_RX = re.compile(r'"message"\s*:\s*"([^"]{1,500})"')
_TORCHELASTIC_TIMESTAMP_RX = re.compile(r'"timestamp"\s*:\s*(\d{9,})')
_PROCESS_FAILURE_RX = re.compile(
    r"ProcessFailure\(\s*local_rank\s*[=:]?\s*(\d+),\s*pid\s*[=:]?\s*(\d+),\s*exitcode\s*[=:]?\s*(-?\d+),\s*error_file\s*[=:]?\s*([^\)\s]+)"
)
_GENERIC_EXIT_CODE_RX = re.compile(r"(?i)\bexit code[: ]+(-?\d+)\b|\(exit code\s+(-?\d+)\)")
_SLURM_EXIT_CODE_RX = re.compile(r"(?i)\bExitCode\b\s*[=:]\s*(-?\d+):(\d+)")
_SLURM_DERIVED_EXIT_CODE_RX = re.compile(r"(?i)\bDerivedExitCode\b\s*[=:]\s*(-?\d+):(\d+)")
_SLURM_STATE_RX = re.compile(r"(?i)\b(?:JobState|State)\b\s*[=:]\s*([A-Z_]+)")
_SLURM_REASON_RX = re.compile(r"(?i)\bReason\b\s*[=:]\s*([A-Za-z0-9_./:-]+)")
_SLURM_NODELIST_RX = re.compile(r"(?i)\b(?:NodeList|BatchHost)\b\s*[=:]\s*([A-Za-z0-9_,\[\].-]+)")
_SLURM_JOBID_INLINE_RX = re.compile(r"(?i)\bJobID\b\s*[=:]\s*([A-Za-z0-9._-]+)")
_ALLOCATION_ID_RX = re.compile(r"(?i)\bAllocation id:\s*([A-Za-z0-9.-]+)|\bid:\s*([A-Za-z0-9.-]{12,}\.\d+)")
_TASK_TYPE_RX = re.compile(r"(?i)\bTASK_TYPE_[A-Z_]+\b")
_RESOURCE_POOL_RX = re.compile(r"(?i)\bResource pool:\s*([A-Za-z0-9_.-]+)")
_AGENT_ID_RX = re.compile(r"(?i)\bagent_id=([A-Za-z0-9_.-]+)")
_VISIBLE_GPU_UUID_RX = re.compile(r"GPU-[0-9a-fA-F-]{8,}")
_SERVICE_PORT_RX = re.compile(r"(?i)\bServer listening on\s+(?:[^\s]+\s+)?port\s+(\d+)\.")
_ACTIVATE_MISSING_RX = re.compile(r"(?i)line\s+\d+:\s+([^\s:]+(?:/bin/activate|activate)):\s+No such file or directory")
_RANK_LOCAL_RANK_RX = re.compile(r"(?i)\[Rank\s+(\d+)\].*?Local rank:\s*(\d+)")


# ============================
# Core deterministic logic
# ============================

def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def redact_secrets(text: str) -> str:
    out = text or ""
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


def _non_empty_lines(text: str) -> list[str]:
    return [line.rstrip() for line in (text or "").splitlines() if line.strip()]


def _display_line(line_no: int, line: str) -> str:
    return f"L{line_no}: {line}"


def _build_log_preview(lines: list[str], suspicious: list[str], max_lines: int = MAX_PREVIEW_LINES) -> tuple[str, bool]:
    if len(lines) <= max_lines:
        return "\n".join(lines[:max_lines]), False

    head_count = min(HEAD_PREVIEW_LINES, max_lines)
    tail_count = min(TAIL_PREVIEW_LINES, max_lines - head_count)
    head = lines[:head_count]
    tail = lines[-tail_count:] if tail_count else []

    preview_lines = list(head)
    preview_lines.append(
        f"... preview truncated: showing {head_count} head lines and {tail_count} tail lines out of {len(lines)} total lines ..."
    )

    if suspicious:
        preview_lines.append("... key signal lines ...")
        preview_lines.extend(suspicious[: min(len(suspicious), max_lines - len(preview_lines))])

    if tail:
        preview_lines.append("... tail lines ...")
        remaining = max_lines - len(preview_lines)
        preview_lines.extend(tail[:remaining] if remaining > 0 else [])

    return "\n".join(preview_lines[:max_lines]), True


def _counter_to_list(counter: Counter, limit: int = 10) -> list[dict]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _extract_first_match(line: str, patterns: list[re.Pattern]) -> str | None:
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def _clean_host_candidate(candidate: str | None, line: str) -> str | None:
    if not candidate:
        return None
    cleaned = candidate.strip().strip("[]()")
    lowered = cleaned.lower()
    if lowered in _HOST_STOPWORDS:
        return None
    if re.search(r"(?i)\bhost key\b", line):
        return None
    if "." not in cleaned and "-" not in cleaned and not re.search(r"\d", cleaned) and len(cleaned) <= 6:
        return None
    return cleaned


def _clean_job_candidate(candidate: str | None) -> str | None:
    if not candidate:
        return None
    cleaned = candidate.strip().strip("[](),")
    lowered = cleaned.lower()
    if not cleaned or lowered in {"job", "jobs", "unknown"}:
        return None
    if "/" in cleaned or "\\" in cleaned:
        return None
    if re.search(r"(?i)\.(?:png|jpe?g|gif|svg|txt|log|out|err|json|ya?ml|pth|pt|sbatch|sh)\b", cleaned):
        return None
    return cleaned


def analyze_log_structure(text: str) -> dict:
    clean = redact_secrets(text)
    indexed_lines = [(idx + 1, line) for idx, line in enumerate(_non_empty_lines(clean))]

    shell_prompts = []
    embedded_artifacts = []
    source_files = []

    for line_no, line in indexed_lines:
        if _SOURCE_HEADER_RX.match(line):
            source_files.append(line)
        if any(pattern.search(line) for pattern in _SHELL_PROMPT_PATTERNS):
            shell_prompts.append(_display_line(line_no, line))
            cat_match = _CAT_COMMAND_RX.search(line)
            if cat_match:
                embedded_artifacts.append(
                    {
                        "line": line_no,
                        "path": cat_match.group(1),
                    }
                )

    mixed_sources = bool(shell_prompts and (embedded_artifacts or source_files))

    return {
        "has_shell_prompts": bool(shell_prompts),
        "shell_prompts": shell_prompts[:5],
        "embedded_artifacts": embedded_artifacts[:10],
        "source_files": source_files[:10],
        "mixed_sources": mixed_sources,
    }


def _supports_browser_investigation(profile: str, topology: dict, grounding_score: float, signatures: dict) -> bool:
    if profile in {"access_ops", "generic_ops"}:
        return False
    if profile == "platform_ops":
        platform = signatures.get("platform", {})
        return bool(platform.get("allocation_ids") or platform.get("resource_pools") or topology.get("hosts"))
    if grounding_score < 0.65:
        return False
    return bool(topology.get("hosts") or topology.get("job_ids") or topology.get("network_interfaces"))


def _supports_telemetry_investigation(topology: dict, grounding_score: float) -> bool:
    if grounding_score < 0.65:
        return False
    return bool(topology.get("hosts") or topology.get("network_interfaces"))


def assess_incident_scope(text: str, env_context: str = "") -> dict:
    clean = redact_secrets(text)
    lower = clean.lower()
    env_lower = (env_context or "").lower()

    runtime_hits = []
    runtime_score = 0.0
    for label, pattern, weight in _HPC_SCOPE_HINTS:
        if pattern.search(clean):
            runtime_hits.append(label)
            runtime_score += weight

    platform_hits = []
    platform_score = 0.0
    for label, pattern, weight in _HPC_PLATFORM_HINTS:
        if pattern.search(clean):
            platform_hits.append(label)
            platform_score += weight

    non_hpc_hits = []
    non_hpc_score = 0.0
    for label, pattern, weight in _NON_HPC_SCOPE_HINTS:
        if pattern.search(clean):
            non_hpc_hits.append(label)
            non_hpc_score += weight

    env_bonus = 0.0
    if (runtime_hits or platform_hits) and re.search(
        r"(?i)\bcluster\b|\bslurm\b|\bnccl\b|\bgpu\b|\binfiniband\b|\bddp\b|\bresource pool\b|\bcontainer\b",
        env_lower,
    ):
        env_bonus = 0.06
    runtime_score = min(1.0, runtime_score + (env_bonus if runtime_hits else 0.0))
    platform_score = min(1.0, platform_score + (env_bonus if platform_hits else 0.0))
    non_hpc_score = min(1.0, non_hpc_score)
    hpc_score = round(max(runtime_score, platform_score), 2)

    if non_hpc_score >= 0.45 and hpc_score < 0.35:
        scope = "access_or_bootstrap_issue"
        should_enable_hpc = False
        workflow_profile = "access_ops"
        explanation = "The log is dominated by SSH/auth/bootstrap markers rather than job-runtime failure evidence."
    elif platform_score >= 0.42 and platform_score >= runtime_score + 0.05:
        scope = "hpc_platform_incident"
        should_enable_hpc = True
        workflow_profile = "platform_ops"
        explanation = "The log contains strong platform/orchestrator signals for a cluster task, shell service, or container lifecycle incident."
    elif runtime_score >= 0.45 and runtime_score >= non_hpc_score + 0.1:
        scope = "hpc_runtime_incident"
        should_enable_hpc = True
        workflow_profile = "runtime_ops"
        explanation = "The log contains strong runtime markers for distributed training or HPC infrastructure failure."
    elif hpc_score >= 0.3:
        scope = "possible_hpc_incident"
        should_enable_hpc = True
        workflow_profile = "platform_ops" if platform_score >= runtime_score else "runtime_ops"
        explanation = "The log shows some HPC-runtime markers, but the evidence is incomplete or mixed."
    else:
        scope = "generic_system_issue"
        should_enable_hpc = False
        workflow_profile = "generic_ops"
        explanation = "The log does not currently contain enough HPC-runtime evidence to justify GPU/fabric-oriented workflows."

    if "ssh -vvv" in lower and "permission denied" in lower:
        explanation += " This specific sample looks like an SSH authentication or proxy-jump failure."
    if workflow_profile == "platform_ops" and "user requested kill" in lower and "exit code 137" in lower:
        explanation += " The exit code 137 here lines up with an explicit kill/SIGKILL event and should not be treated as OOM by default."
    if workflow_profile == "platform_ops" and "service of " in lower and " is available" in lower and "accepted publickey" in lower:
        explanation += " The interactive service appears to have come up successfully, so the sshd pid/login-record warnings may be secondary non-root container side effects."

    return {
        "scope": scope,
        "should_enable_hpc_workflows": should_enable_hpc,
        "workflow_profile": workflow_profile,
        "hpc_relevance_score": hpc_score,
        "runtime_hpc_score": round(runtime_score, 2),
        "platform_hpc_score": round(platform_score, 2),
        "non_hpc_score": round(non_hpc_score, 2),
        "hpc_markers": _unique_keep_order(runtime_hits + platform_hits),
        "platform_markers": _unique_keep_order(platform_hits),
        "non_hpc_markers": _unique_keep_order(non_hpc_hits),
        "explanation": explanation,
    }


def _score_signal_line(line: str, line_no: int, total_lines: int) -> float:
    score = 0.0

    for pattern, weight in _STRONG_SIGNAL_HINTS:
        if pattern.search(line):
            score += weight

    if _SIGNAL_PATTERN.search(line):
        score += 2.5

    if any(pattern.search(line) for pattern in _LOW_VALUE_SIGNAL_HINTS):
        score -= 4.0

    if re.match(r"(?i)^debug\d:", line):
        score -= 0.75

    if re.search(r"(?i)\bnccl\b|\bslurm\b|\btorchelastic\b|oom|xid|permission denied|publickey|password:", line):
        score += 1.0

    if re.search(r"(?i)\bnccl\b.*\binfo\b", line):
        score -= 2.0
    if re.search(r"(?i)\bchannel \d+/\d+\b|\bring \d+\b|\btree \d+\b|\bConnected all rings\b", line):
        score -= 3.0
    if re.search(r"(?i)\bproxyProgressAsync\b|\bsendConnect\b|\brecvConnect\b|\bIb Alloc Size\b|\bCuda Host Alloc Size\b", line):
        score -= 2.0

    # Prefer the first anchored failure over later cascading info/noise.
    score += max(0.0, 2.0 * (1.0 - ((line_no - 1) / max(1, total_lines))))
    return score


def _select_top_signals(indexed_lines: list[tuple[int, str]], limit: int = MAX_SIGNAL_LINES) -> list[str]:
    scored = []
    total_lines = max(1, len(indexed_lines))
    for line_no, line in indexed_lines:
        score = _score_signal_line(line, line_no, total_lines)
        if score >= 3.5:
            scored.append((score, line_no, line))

    if not scored:
        fallback_tail = indexed_lines[-8:] if len(indexed_lines) > 8 else indexed_lines
        return [_display_line(line_no, line) for line_no, line in fallback_tail]

    scored.sort(key=lambda item: (-item[0], item[1]))
    top = []
    seen_templates = set()
    for score, line_no, line in scored:
        template = normalize_log_template(line)
        if template in seen_templates:
            continue
        top.append((score, line_no, line))
        seen_templates.add(template)
        if len(top) >= limit:
            break
    top.sort(key=lambda item: item[1])
    return [_display_line(line_no, line) for _, line_no, line in top]


def normalize_log_template(line: str) -> str:
    normalized = line.strip()
    substitutions = [
        (r"\b20\d{2}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}\b", "<TIMESTAMP>"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>"),
        (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>"),
        (r"\b0x[0-9a-fA-F]+\b", "<HEX>"),
        (r"(?<![A-Za-z0-9])/(?:[\w.\-]+/)+[\w.\-]+", "<PATH>"),
        (r"(?<![A-Za-z0-9])[A-Za-z]:\\\\(?:[\w.\-]+\\\\)+[\w.\-]+", "<PATH>"),
        (r"\b\d+\b", "<NUM>"),
    ]
    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:240]


def mine_log_templates(text: str, top_k: int = TOP_TEMPLATE_COUNT) -> dict:
    clean = redact_secrets(text)
    raw_lines = _non_empty_lines(clean)
    if not raw_lines:
        return {
            "total_lines": 0,
            "unique_templates": 0,
            "top_templates": [],
            "rare_signal_lines": [],
        }

    template_counts: Counter = Counter()
    template_examples: dict[str, str] = {}
    line_templates: list[tuple[str, str]] = []

    for line in raw_lines:
        template = normalize_log_template(line)
        template_counts[template] += 1
        template_examples.setdefault(template, line)
        line_templates.append((line, template))

    rare_signal_lines = []
    for idx, (line, template) in enumerate(line_templates, start=1):
        if _SIGNAL_PATTERN.search(line) and template_counts[template] == 1:
            rare_signal_lines.append(_display_line(idx, line))

    top_templates = []
    for template, count in template_counts.most_common(top_k):
        top_templates.append(
            {
                "template": template,
                "count": count,
                "share_pct": round((count / len(raw_lines)) * 100, 2),
                "example": template_examples[template],
            }
        )

    return {
        "total_lines": len(raw_lines),
        "unique_templates": len(template_counts),
        "top_templates": top_templates,
        "rare_signal_lines": rare_signal_lines[:5],
    }


def build_incident_topology(text: str) -> dict:
    clean = redact_secrets(text)
    indexed_lines = [(idx + 1, line) for idx, line in enumerate(_non_empty_lines(clean))]

    sources: Counter = Counter()
    hosts: Counter = Counter()
    ranks: Counter = Counter()
    gpus: Counter = Counter()
    jobs: Counter = Counter()
    interfaces: Counter = Counter()
    host_rank_pairs: Counter = Counter()
    notable_events: list[str] = []
    timestamps: list[str] = []

    for line_no, line in indexed_lines:
        source_match = _SOURCE_HEADER_RX.match(line)
        if source_match:
            sources[source_match.group(1)] += 1

        timestamp_match = _TIMESTAMP_RX.search(line)
        if timestamp_match:
            timestamps.append(timestamp_match.group(1))

        host = _clean_host_candidate(_extract_first_match(line, _HOST_PATTERNS), line)
        rank = _extract_first_match(line, _RANK_PATTERNS)
        gpu = _extract_first_match(line, _GPU_PATTERNS)
        job = _clean_job_candidate(_extract_first_match(line, _JOB_PATTERNS))

        if host:
            hosts[host] += 1
        if rank:
            ranks[f"rank {rank}"] += 1
        if gpu:
            gpus[f"gpu {gpu}"] += 1
        if job:
            jobs[job] += 1
        if host and rank:
            host_rank_pairs[f"{host} / rank {rank}"] += 1

        for match in _INTERFACE_PATTERN.findall(line):
            interfaces[match] += 1

        if len(notable_events) < 10 and re.search(
            r"(?i)\btimeout\b|\brefused\b|stale file handle|xid|oom|permission denied|segfault|traceback|modulenotfounderror|importerror|no such file or directory|user requested kill|non-zero exit code|exited with exit code|login records by non-root|pid file",
            line,
        ):
            notable_events.append(_display_line(line_no, line))

    return {
        "source_files": _counter_to_list(sources),
        "hosts": _counter_to_list(hosts),
        "ranks": _counter_to_list(ranks),
        "gpus": _counter_to_list(gpus),
        "job_ids": _counter_to_list(jobs),
        "network_interfaces": _counter_to_list(interfaces),
        "host_rank_pairs": _counter_to_list(host_rank_pairs),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "notable_events": notable_events,
    }


def _memory_record_from_text(text: str, env_context: str = "") -> dict:
    incident = extract_incident(text)
    templates = mine_log_templates(text)
    topology = build_incident_topology(text)
    return {
        "id": incident["id"],
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "env_context": env_context.strip()[:400],
        "tags": incident["tags"],
        "top_signals": incident["top_signals"][:8],
        "top_templates": [entry["template"] for entry in templates["top_templates"][:5]],
        "hosts": [entry["value"] for entry in topology["hosts"][:5]],
        "preview": incident["redacted_preview"][:800],
    }


def _read_incident_memory() -> list[dict]:
    if not INCIDENT_MEMORY_PATH.exists():
        return []
    records = []
    for line in INCIDENT_MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _signal_token_set(lines: list[str]) -> set[str]:
    joined = " ".join(lines).lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", joined))
    return {token for token in tokens if token not in _MEMORY_STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _incident_similarity(current: dict, previous: dict) -> float:
    tag_score = _jaccard(set(current["tags"]), set(previous.get("tags", [])))
    signal_score = _jaccard(_signal_token_set(current["top_signals"]), _signal_token_set(previous.get("top_signals", [])))
    template_score = _jaccard(set(current["top_templates"]), set(previous.get("top_templates", [])))
    host_score = _jaccard(set(current.get("hosts", [])), set(previous.get("hosts", [])))
    return round((0.4 * tag_score) + (0.3 * signal_score) + (0.2 * template_score) + (0.1 * host_score), 3)


def get_official_guidance(tags: list[str]) -> list[dict]:
    seen_urls = set()
    guidance = []
    for tag in tags:
        for item in OFFICIAL_HPC_GUIDANCE.get(tag, []):
            if item["url"] in seen_urls:
                continue
            guidance.append(item)
            seen_urls.add(item["url"])
    return guidance


def _expected_artifacts_for_tags(tags: list[str]) -> list[dict]:
    artifacts = []
    if "NCCL" in tags or "InfiniBand" in tags or "Network" in tags:
        artifacts.extend(
            [
                {"artifact": "Per-rank logs from all failing ranks", "why": "Distributed failures are often asymmetric across ranks."},
                {"artifact": "Interface/fabric health outputs", "why": "NCCL and RDMA failures need NIC/link-level confirmation."},
            ]
        )
    if "GPU" in tags or "OOM" in tags:
        artifacts.extend(
            [
                {"artifact": "nvidia-smi and GPU topology output", "why": "Confirms device state, topology, and memory pressure."},
                {"artifact": "dmesg Xid excerpts", "why": "Helps distinguish OOM from driver or hardware faults."},
            ]
        )
    if "Scheduler" in tags:
        artifacts.append(
            {"artifact": "Scheduler metadata (sacct/scontrol/job output)", "why": "Needed for exit codes, node list, and retry/failure history."}
        )
    if "TorchElastic" in tags:
        artifacts.append(
            {"artifact": "TorchElastic error.json or error file outputs", "why": "The first worker failure is often recorded there before cascades hide it."}
        )
    if "Disk/FS" in tags:
        artifacts.append(
            {"artifact": "Mount and filesystem health output", "why": "Confirms stale handles, quota issues, or read-only state."}
        )
    if not artifacts:
        artifacts.append(
            {"artifact": "Sibling logs, recent system logs, and environment metadata", "why": "Unclassified failures need broader surrounding evidence."}
        )
    return artifacts


def _unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _parse_exit_code_pairs(matches: list[tuple[str, str]]) -> list[dict]:
    return [
        {
            "raw": f"{exit_code}:{signal}",
            "exit_code": int(exit_code),
            "signal": int(signal),
        }
        for exit_code, signal in matches
    ]


def _parse_generic_exit_codes(text: str) -> list[int]:
    values = []
    for match in _GENERIC_EXIT_CODE_RX.finditer(text):
        raw_value = match.group(1) or match.group(2)
        if raw_value is None:
            continue
        try:
            values.append(int(raw_value))
        except ValueError:
            continue
    return _unique_keep_order(values)


def suggest_playwright_surfaces(tags: list[str]) -> list[str]:
    surfaces = []
    if "Platform" in tags:
        surfaces.extend(["Determined or platform UI", "Cluster job portal"])
    if any(tag in tags for tag in ["GPU", "NCCL", "Network", "InfiniBand"]):
        surfaces.append("Grafana dashboard")
    if "Scheduler" in tags:
        surfaces.extend(["Slurm or scheduler web UI", "Cluster job portal"])
    if "Disk/FS" in tags:
        surfaces.append("Storage or fabric dashboard")
    surfaces.append("Kibana or log search UI")
    return _unique_keep_order(surfaces)


def parse_hpc_failure_artifacts(text: str) -> dict:
    clean = redact_secrets(text)
    lines = _non_empty_lines(clean)
    topology = build_incident_topology(clean)

    process_failures = []
    for match in _PROCESS_FAILURE_RX.finditer(clean):
        process_failures.append(
            {
                "local_rank": int(match.group(1)),
                "pid": int(match.group(2)),
                "exit_code": int(match.group(3)),
                "error_file": match.group(4),
            }
        )

    torchelastic_messages = _unique_keep_order(_TORCHELASTIC_MESSAGE_RX.findall(clean))
    torchelastic_error_files = _unique_keep_order(_TORCHELASTIC_ERROR_FILE_RX.findall(clean))
    torchelastic_timestamps = [int(value) for value in _TORCHELASTIC_TIMESTAMP_RX.findall(clean)]
    torchelastic_detected = any(
        marker in clean
        for marker in ["TORCHELASTIC_ERROR_FILE", "ChildFailedError", "ProcessFailure(", "torch.distributed.elastic"]
    )
    torchelastic_follow_up = []
    if torchelastic_detected:
        torchelastic_follow_up.extend(
            [
                "Collect the first worker error file referenced by TORCHELASTIC_ERROR_FILE.",
                "Compare the earliest worker failure timestamp against scheduler and NCCL timelines.",
                "Keep per-rank stderr/stdout together because the first failing worker often explains later cascading errors.",
            ]
        )
    if not torchelastic_error_files and torchelastic_detected:
        torchelastic_follow_up.append("Search the run directory for error.json or TORCHELASTIC error files.")

    slurm_exit_codes = _parse_exit_code_pairs(_SLURM_EXIT_CODE_RX.findall(clean))
    slurm_derived_exit_codes = _parse_exit_code_pairs(_SLURM_DERIVED_EXIT_CODE_RX.findall(clean))
    slurm_states = _unique_keep_order(_SLURM_STATE_RX.findall(clean))
    slurm_reasons = _unique_keep_order(
        match.group(1)
        for line in lines
        if re.search(r"(?i)\bslurm\b|slurmstepd|sacct|scontrol|JobState|NodeList|BatchHost", line)
        for match in [_SLURM_REASON_RX.search(line)]
        if match
    )
    slurm_nodelists = _unique_keep_order(_SLURM_NODELIST_RX.findall(clean))
    slurm_job_ids = _unique_keep_order(_SLURM_JOBID_INLINE_RX.findall(clean))
    slurm_job_ids.extend([entry["value"] for entry in topology["job_ids"]])
    slurm_job_ids = _unique_keep_order(slurm_job_ids)
    slurm_detected = any(
        marker in clean.lower()
        for marker in ["slurm", "slurmstepd", "srun", "sbatch", "sacct", "scontrol"]
    ) or bool(slurm_exit_codes or slurm_derived_exit_codes or slurm_job_ids)
    slurm_follow_up = []
    if slurm_detected:
        slurm_follow_up.extend(
            [
                "Query sacct with ExitCode, DerivedExitCode, Comment, and NodeList for the failing job.",
                "Use scontrol show job to confirm node allocation, batch host, and failure reason.",
            ]
        )
        if slurm_derived_exit_codes:
            slurm_follow_up.append("Treat DerivedExitCode as a stronger clue than a 0:0 batch script exit when steps failed underneath.")

    nccl_env = [
        {"name": name, "value": value}
        for name, value in _NCCL_ENV_RX.findall(clean)
    ]
    nccl_env_names = [entry["name"] for entry in nccl_env]
    nccl_transport_lines = [
        _display_line(idx, line)
        for idx, line in enumerate(lines, start=1)
        if re.search(r"(?i)\bnccl\b|\bnet/ib\b|\broce\b|\brdma\b|\bibstat\b|\bibv_\w+\b|\bmlx5\b|\bNCCL_[A-Z0-9_]+\b", line)
    ][:10]
    nccl_detected = any("nccl" in line.lower() for line in lines)
    nccl_ras_related = any(name.startswith("NCCL_RAS_") for name in nccl_env_names) or "RAS" in clean
    nccl_follow_up = []
    if nccl_detected:
        nccl_follow_up.extend(
            [
                "Capture the effective NCCL environment (NCCL_DEBUG, NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_IB_DISABLE, NCCL_RAS_ENABLE).",
                "Correlate NCCL transport lines with interface health and per-rank failures.",
            ]
        )
        if not nccl_ras_related:
            nccl_follow_up.append("If your NCCL version supports it, enable or query NCCL RAS to get communicator-level state.")

    allocation_ids = _unique_keep_order(
        (match.group(1) or match.group(2))
        for match in _ALLOCATION_ID_RX.finditer(clean)
        if (match.group(1) or match.group(2))
    )
    task_types = _unique_keep_order(_TASK_TYPE_RX.findall(clean))
    resource_pools = _unique_keep_order(_RESOURCE_POOL_RX.findall(clean))
    agent_ids = _unique_keep_order(_AGENT_ID_RX.findall(clean))
    visible_gpu_uuids = _unique_keep_order(_VISIBLE_GPU_UUID_RX.findall(clean))
    service_ports = _unique_keep_order(_SERVICE_PORT_RX.findall(clean))
    generic_exit_codes = _parse_generic_exit_codes(clean)
    ssh_service_warnings = [
        _display_line(idx, line)
        for idx, line in enumerate(lines, start=1)
        if re.search(r"(?i)Couldn'?t create pid file|Attempt to write login records by non-root user", line)
    ][:10]
    service_ready_lines = [
        _display_line(idx, line)
        for idx, line in enumerate(lines, start=1)
        if re.search(r"(?i)Service of .* is available|Accepted publickey for ", line)
    ][:10]
    lifecycle_lines = [
        _display_line(idx, line)
        for idx, line in enumerate(lines, start=1)
        if re.search(
            r"(?i)Restoring .*|forcibly killing allocation|resources failed with non-zero exit code|allocation killed after all resources exited|Running task container",
            line,
        )
    ][:12]
    platform_detected = any(
        [
            "determined" in clean.lower(),
            allocation_ids,
            task_types,
            resource_pools,
            agent_ids,
            visible_gpu_uuids,
        ]
    )
    user_requested_kill = "user requested kill" in clean.lower()
    service_ready = bool(service_ready_lines)
    if user_requested_kill and 137 in generic_exit_codes:
        termination_interpretation = "user_requested_sigkill"
    elif 137 in generic_exit_codes:
        termination_interpretation = "sigkill_or_possible_oom"
    elif generic_exit_codes:
        termination_interpretation = "non_zero_exit"
    else:
        termination_interpretation = "none_detected"

    platform_follow_up = []
    if platform_detected:
        platform_follow_up.extend(
            [
                "Inspect the platform task or allocation event history before assuming an application-level failure.",
                "Check startup hooks, container entrypoint, and sshd configuration for non-root interactive shell behavior.",
                "Compare service-ready lines against sshd warnings to decide whether those warnings were fatal or cosmetic.",
            ]
        )
        if user_requested_kill:
            platform_follow_up.append("Confirm who initiated the kill request before treating exit code 137 as an infrastructure fault.")
        elif 137 in generic_exit_codes:
            platform_follow_up.append("Differentiate SIGKILL or OOM from orchestrator policy by checking platform events and kernel memory signals.")

    return {
        "torchelastic": {
            "detected": torchelastic_detected,
            "error_files": torchelastic_error_files,
            "failure_timestamps_epoch": torchelastic_timestamps,
            "messages": torchelastic_messages[:5],
            "process_failures": process_failures[:10],
            "recommended_follow_up": torchelastic_follow_up,
        },
        "slurm": {
            "detected": slurm_detected,
            "job_ids": slurm_job_ids,
            "states": slurm_states,
            "reasons": slurm_reasons[:5],
            "nodelists": slurm_nodelists[:5],
            "exit_codes": slurm_exit_codes[:10],
            "derived_exit_codes": slurm_derived_exit_codes[:10],
            "recommended_follow_up": slurm_follow_up,
        },
        "nccl": {
            "detected": nccl_detected,
            "ras_related": nccl_ras_related,
            "env": nccl_env[:20],
            "transport_lines": nccl_transport_lines,
            "recommended_follow_up": nccl_follow_up,
        },
        "platform": {
            "detected": platform_detected,
            "platform": "Determined/HPE MLDE" if "determined" in clean.lower() else None,
            "allocation_ids": allocation_ids[:10],
            "task_types": task_types[:10],
            "resource_pools": resource_pools[:10],
            "agent_ids": agent_ids[:10],
            "visible_gpu_uuids": visible_gpu_uuids[:12],
            "service_ports": service_ports[:6],
            "service_ready": service_ready,
            "user_requested_kill": user_requested_kill,
            "generic_exit_codes": generic_exit_codes[:10],
            "termination_interpretation": termination_interpretation,
            "ssh_service_warnings": ssh_service_warnings,
            "service_ready_lines": service_ready_lines,
            "lifecycle_lines": lifecycle_lines,
            "recommended_follow_up": platform_follow_up,
        },
    }


def build_collection_manifest(text: str, env_context: str = "") -> dict:
    incident = extract_incident(text)
    scope = incident["scope_assessment"]
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")
    topology = build_incident_topology(text)
    signatures = parse_hpc_failure_artifacts(text)
    gaps = assess_evidence_gaps(text, env_context)
    guidance = get_official_guidance(incident["tags"])

    job_ids = [entry["value"] for entry in topology["job_ids"][:3]]
    interfaces = [entry["value"] for entry in topology["network_interfaces"][:3]]
    hosts = [entry["value"] for entry in topology["hosts"][:3]]
    ranks = [entry["value"] for entry in topology["ranks"][:6]]
    surfaces = suggest_playwright_surfaces(incident["tags"])
    browser_ready = _supports_browser_investigation(profile, topology, gaps["grounding_score"], signatures)
    telemetry_ready = _supports_telemetry_investigation(topology, gaps["grounding_score"])

    if profile == "access_ops":
        return {
            "incident_id": incident["id"],
            "tags": incident["tags"],
            "scope_assessment": scope,
            "ready_for_llm": False,
            "grounding_score": gaps["grounding_score"],
            "confidence_label": gaps["confidence_label"],
            "expected_artifacts": [
                {
                    "artifact": "Resolved SSH config and login path",
                    "why": "This sample looks like a cluster access/bootstrap problem, not a runtime GPU/fabric failure.",
                },
                {
                    "artifact": "Actual runtime logs from inside the cluster",
                    "why": "GPU/fabric workflows should start only after the access path succeeds and real job logs are available.",
                },
            ],
            "filesystem": [
                {
                    "priority": "high",
                    "artifact": "SSH config and known_hosts",
                    "path_hints": ["~/.ssh/config", "~/.ssh/known_hosts"],
                    "why": "ProxyJump, username, and key mismatches are common causes of cluster access failures.",
                }
            ],
            "shell": [
                {
                    "priority": "high",
                    "command": "ssh -G <cluster_alias>",
                    "why": "Inspect the resolved SSH config, hostname, user, and jump settings.",
                },
                {
                    "priority": "high",
                    "command": "ssh -vvv <cluster_alias>",
                    "why": "Confirm where authentication fails before moving to runtime diagnostics.",
                },
            ],
            "docs": [],
            "telemetry": [],
            "playwright": [],
            "structured_signatures": signatures,
        }

    if profile == "generic_ops":
        return {
            "incident_id": incident["id"],
            "tags": incident["tags"],
            "scope_assessment": scope,
            "ready_for_llm": False,
            "grounding_score": gaps["grounding_score"],
            "confidence_label": gaps["confidence_label"],
            "expected_artifacts": [
                {
                    "artifact": "Adjacent logs and process-level context",
                    "why": "This sample is not clearly classified yet, so surrounding evidence matters more than domain-specific assumptions.",
                },
                {
                    "artifact": "Recent system and runtime state from the affected environment",
                    "why": "Exit codes, kernel messages, and process state help separate app failures from host/container issues.",
                },
            ],
            "filesystem": [
                {
                    "priority": "high",
                    "artifact": "Sibling logs, configs, and launch scripts",
                    "path_hints": ["**/*.log", "**/*.out", "**/*.err", "**/*.yaml", "**/*.sh"],
                    "why": "Nearby artifacts often explain ambiguous single-log failures.",
                }
            ],
            "shell": [
                {
                    "priority": "high",
                    "command": "ps -ef | head -200 && ss -tulpn | head -100",
                    "why": "Inspect process and listener state near the failure.",
                },
                {
                    "priority": "medium",
                    "command": "df -h && free -h || true && dmesg | tail -200 || true",
                    "why": "Check whether storage, memory, or kernel events explain the failure class.",
                },
            ],
            "docs": [],
            "telemetry": [],
            "playwright": [],
            "structured_signatures": signatures,
        }

    if profile == "platform_ops":
        platform = signatures["platform"]
        return {
            "incident_id": incident["id"],
            "tags": incident["tags"],
            "scope_assessment": scope,
            "ready_for_llm": gaps["ready_for_llm"],
            "grounding_score": gaps["grounding_score"],
            "confidence_label": gaps["confidence_label"],
            "expected_artifacts": [
                {
                    "artifact": "Platform allocation/task event history",
                    "why": "Allocation restores, shell-service availability, and kill decisions are orchestrator events, not just app logs.",
                },
                {
                    "artifact": "Startup hooks, container entrypoint, and sshd/service config",
                    "why": "Non-root shell containers often emit warnings that are harmless unless they block readiness or user access.",
                },
            ],
            "filesystem": [
                {
                    "priority": "high",
                    "artifact": "Task startup assets and platform hooks",
                    "path_hints": ["/run/determined/**", "startup-hook.sh", "/run/determined/dynamic-tcd-startup-hook.sh"],
                    "why": "These files explain how the interactive service and container startup were configured.",
                },
                {
                    "priority": "high",
                    "artifact": "sshd config, pid paths, and runtime directories",
                    "path_hints": ["/etc/ssh/sshd_config", "/run", "/var/run"],
                    "why": "The pid-file and login-record warnings point at how sshd is being launched inside a non-root container.",
                },
                {
                    "priority": "medium",
                    "artifact": "Platform task/allocation event exports or agent/master logs",
                    "path_hints": ["determined-master*.log", "determined-agent*.log", "task-events*.json"],
                    "why": "These artifacts reveal whether restores and kills were operator-driven, policy-driven, or infrastructure-driven.",
                },
            ],
            "shell": [
                {
                    "priority": "high",
                    "command": "id && whoami && ps -ef | egrep 'sshd|determined|bash|python' || true",
                    "why": "Verify the container user and the interactive service process tree.",
                },
                {
                    "priority": "high",
                    "command": "ss -tulpn | egrep '3282|22' || true && ls -ld /run /var/run /run/determined || true",
                    "why": "Check whether the shell service is bound as expected and whether pid/runtime directories are writable.",
                },
                {
                    "priority": "medium",
                    "command": "env | egrep '^(DET|HOSTNAME|USER|HOME|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)=' || true",
                    "why": "Capture platform and container environment that shaped GPU visibility and service behavior.",
                },
                {
                    "priority": "medium",
                    "command": "cat /etc/ssh/sshd_config || true && grep -R 'PidFile\\|UsePAM' /etc/ssh 2>/dev/null || true",
                    "why": "Confirm whether sshd is trying to use root-owned paths or login-record features that do not fit a non-root shell container.",
                },
            ],
            "docs": [
                {
                    "priority": "medium",
                    "title": item["title"],
                    "url": item["url"],
                }
                for item in guidance
            ],
            "telemetry": [
                {
                    "priority": "low" if platform["user_requested_kill"] else "medium",
                    "query": "Platform task events, restore history, and kill initiator around the allocation timeline",
                    "filters": platform["allocation_ids"][:3] or hosts or ["current allocation"],
                    "why": "Confirms whether the platform, an operator, or a resource policy drove the lifecycle transition.",
                }
            ],
            "playwright": [
                {
                    "priority": "medium",
                    "surface": "Determined or platform UI",
                    "filters": {
                        "allocation_ids": platform["allocation_ids"][:3],
                        "resource_pools": platform["resource_pools"][:3],
                        "task_types": platform["task_types"][:4],
                        "hosts": hosts,
                    },
                    "goal": "Confirm task lifecycle, service readiness, restore history, and kill reason from the platform timeline.",
                },
                {
                    "priority": "low",
                    "surface": "Cluster job portal",
                    "filters": {
                        "hosts": hosts,
                        "resource_pools": platform["resource_pools"][:3],
                    },
                    "goal": "Correlate platform events with node-level visibility or broader cluster operations.",
                },
            ],
            "structured_signatures": signatures,
        }

    filesystem_tasks = [
        {
            "priority": "high",
            "artifact": "Sibling rank logs and stderr/stdout files",
            "path_hints": ["**/*rank*.log", "**/*stderr*", "**/*stdout*"],
            "why": "Distributed failures often surface differently across ranks and worker processes.",
        },
        {
            "priority": "high",
            "artifact": "TorchElastic error files",
            "path_hints": signatures["torchelastic"]["error_files"] or ["**/error.json", "**/*elastic*error*"],
            "why": "PyTorch Elastic writes the first worker failure and its concise message to a structured error file.",
        },
        {
            "priority": "medium",
            "artifact": "Scheduler outputs",
            "path_hints": [f"slurm-{job_id}.out" for job_id in job_ids] or ["slurm-*.out", "slurm-*.err"],
            "why": "Scheduler outputs carry node lists, step failures, exit codes, and timing.",
        },
    ]

    shell_tasks = [
        {
            "priority": "high",
            "command": "env | egrep '^(NCCL|TORCHELASTIC|SLURM)_' || true",
            "why": "Capture the exact runtime knobs that shaped transport, scheduling, and worker failure handling.",
        },
        {
            "priority": "high",
            "command": "nvidia-smi && nvidia-smi topo -m",
            "why": "Confirm GPU state, topology, and memory pressure around the failure.",
        },
        {
            "priority": "medium",
            "command": "dmesg | egrep -i 'xid|mlx|ib|oom|nvlink' | tail -200 || true",
            "why": "Kernel-level GPU, NIC, and OOM signals frequently clarify ambiguous log symptoms.",
        },
    ]
    if incident["tags"] and any(tag in incident["tags"] for tag in ["NCCL", "Network", "InfiniBand"]):
        iface = interfaces[0] if interfaces else "<iface>"
        shell_tasks.extend(
            [
                {
                    "priority": "high",
                    "command": "ibstat || true && ibv_devinfo || true",
                    "why": "Verify RDMA device visibility and fabric health.",
                },
                {
                    "priority": "medium",
                    "command": f"ip a show {iface} || true && ethtool {iface} || true",
                    "why": "Confirm the interface selected by NCCL is healthy and correctly configured.",
                },
            ]
        )
    if signatures["slurm"]["detected"]:
        job_id = signatures["slurm"]["job_ids"][0] if signatures["slurm"]["job_ids"] else "$SLURM_JOB_ID"
        shell_tasks.extend(
            [
                {
                    "priority": "high",
                    "command": f"sacct -X -j {job_id} -o JobID,State,ExitCode,DerivedExitCode,Comment,NodeList",
                    "why": "Gather batch and step-level failure signals directly from Slurm accounting.",
                },
                {
                    "priority": "medium",
                    "command": f"scontrol show job {job_id}",
                    "why": "Recover node allocation, batch host, reason, and other scheduler metadata.",
                },
            ]
        )

    docs_tasks = [
        {
            "priority": "medium",
            "title": item["title"],
            "url": item["url"],
        }
        for item in guidance
    ]

    telemetry_tasks = []
    if telemetry_ready and any(tag in incident["tags"] for tag in ["GPU", "OOM"]):
        telemetry_tasks.append(
            {
                "priority": "medium",
                "query": "GPU Xid/ECC/reset events around the failure window",
                "filters": hosts or ["affected hosts"],
                "why": "GPU-side faults often explain downstream NCCL or OOM noise.",
            }
        )
    if telemetry_ready and any(tag in incident["tags"] for tag in ["NCCL", "Network", "InfiniBand"]):
        telemetry_tasks.append(
            {
                "priority": "medium",
                "query": "Fabric health, link flaps, retransmits, or packet error counters",
                "filters": interfaces or ["cluster fabric"],
                "why": "Transport-level anomalies can be correlated with NCCL warnings and stalled collectives.",
            }
        )
    if signatures["slurm"]["detected"]:
        telemetry_tasks.append(
            {
                "priority": "low",
                "query": "Scheduler events for the failing job and nodes",
                "filters": job_ids or ["current job"],
                "why": "Correlates node drains, requeues, and controller-side issues with the failure timeline.",
            }
        )

    playwright_tasks = []
    if browser_ready:
        playwright_tasks = [
            {
                "priority": "medium",
                "surface": surface,
                "filters": {
                    "hosts": hosts,
                    "job_ids": job_ids,
                    "ranks": ranks[:4],
                    "interfaces": interfaces,
                },
                "goal": "Confirm whether browser-visible operational evidence supports the log-derived hypothesis.",
            }
            for surface in surfaces
        ]

    return {
        "incident_id": incident["id"],
        "tags": incident["tags"],
        "scope_assessment": scope,
        "ready_for_llm": gaps["ready_for_llm"],
        "grounding_score": gaps["grounding_score"],
        "confidence_label": gaps["confidence_label"],
        "expected_artifacts": gaps["expected_artifacts"],
        "filesystem": filesystem_tasks,
        "shell": shell_tasks,
        "docs": docs_tasks,
        "telemetry": telemetry_tasks,
        "playwright": playwright_tasks,
        "log_structure": gaps["log_structure"],
        "structured_signatures": signatures,
    }


def save_incident_to_memory(text: str, env_context: str = "") -> dict:
    if not text.strip():
        return {"status": "error", "message": "No logs provided."}

    records = _read_incident_memory()
    record = _memory_record_from_text(text, env_context)

    if any(existing.get("id") == record["id"] for existing in records):
        return {
            "status": "already_exists",
            "message": f"Incident {record['id']} is already stored in local memory.",
            "incident_id": record["id"],
            "memory_entries": len(records),
        }

    with INCIDENT_MEMORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return {
        "status": "saved",
        "message": f"Stored incident {record['id']} in local memory.",
        "incident_id": record["id"],
        "memory_entries": len(records) + 1,
        "memory_path": str(INCIDENT_MEMORY_PATH),
    }


def find_similar_incidents(text: str, limit: int = 5) -> dict:
    records = _read_incident_memory()
    if not records:
        return {
            "status": "empty",
            "message": "Local incident memory is empty. Save a few incidents first.",
            "memory_entries": 0,
            "matches": [],
        }

    current = _memory_record_from_text(text)
    matches = []
    for record in records:
        if record.get("id") == current["id"]:
            continue
        score = _incident_similarity(current, record)
        if score < MIN_MEMORY_SIMILARITY:
            continue
        matches.append(
            {
                "incident_id": record.get("id"),
                "similarity": score,
                "saved_at": record.get("saved_at"),
                "tags": record.get("tags", []),
                "top_signals": record.get("top_signals", []),
                "hosts": record.get("hosts", []),
                "preview": record.get("preview", ""),
            }
        )

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "status": "ok",
        "message": "Found similar incidents." if matches else "No similar incidents found yet.",
        "memory_entries": len(records),
        "matches": matches[:limit],
    }


def extract_incident(text: str) -> dict:
    clean = redact_secrets(text)
    indexed_lines = [(idx + 1, line) for idx, line in enumerate(_non_empty_lines(clean))]
    display_lines = [_display_line(idx, line) for idx, line in indexed_lines]
    scope = assess_incident_scope(clean)
    signatures = parse_hpc_failure_artifacts(clean)
    suspicious = _select_top_signals(indexed_lines)

    tags = [name for name, rx in _ERROR_HINTS if rx.search(clean)]
    if scope.get("workflow_profile") == "access_ops" and "SSH" in tags and "Network" in tags:
        tags = [tag for tag in tags if tag != "Network"]
    if scope.get("workflow_profile") == "platform_ops" and "Platform" not in tags:
        tags.append("Platform")
    if signatures["platform"]["visible_gpu_uuids"] and "GPU" not in tags:
        tags.append("GPU")
    tags = _unique_keep_order(tags)

    ts_match = _TIMESTAMP_RX.search(clean)
    timestamp = ts_match.group(1) if ts_match else None
    preview, truncated = _build_log_preview(display_lines, suspicious)

    return {
        "id": _fingerprint(clean),
        "timestamp_hint": timestamp,
        "tags": tags or ["Unclassified"],
        "top_signals": suspicious or display_lines[:8],
        "line_count": len(display_lines),
        "truncated_preview": truncated,
        "redacted_preview": preview,
        "scope_assessment": scope,
    }


def make_incident_card(text: str) -> str:
    inc = extract_incident(text)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scope = inc["scope_assessment"]

    # IMPORTANT: Do NOT use triple-backticks here, it breaks your UI rendering.
    parts = []
    parts.append(f"## 🧪 Incident `{inc['id']}`")
    parts.append(f"Generated: {now}  ")
    if inc["timestamp_hint"]:
        parts.append(f"Timestamp hint: `{inc['timestamp_hint']}`  ")
    parts.append(f"Tags: {', '.join(inc['tags'])}\n")
    parts.append(f"Scope: `{scope['scope']}` (HPC relevance `{scope['hpc_relevance_score']}`)\n")

    parts.append("### 🔥 Top Signals")
    for s in inc["top_signals"]:
        parts.append(f"- {html.escape(s)}")

    parts.append("\n### 🧼 Redacted Log Preview")
    if inc["truncated_preview"]:
        parts.append(f"Preview is truncated from `{inc['line_count']}` non-empty lines.")
    parts.append("<pre>")
    parts.append(html.escape(inc["redacted_preview"]))
    parts.append("</pre>")

    return "\n".join(parts)


def build_llm_evidence(text: str, max_lines: int = MAX_PREVIEW_LINES) -> str:
    inc = extract_incident(text)
    templates = mine_log_templates(text)
    topology = build_incident_topology(text)
    gaps = assess_evidence_gaps(text)
    structure = analyze_log_structure(text)
    scope = inc["scope_assessment"]
    parts = [
        f"Incident ID: {inc['id']}",
        f"Tags: {', '.join(inc['tags'])}",
        f"Scope: {scope['scope']}",
        f"Workflow profile: {scope.get('workflow_profile', 'unknown')}",
        f"HPC relevance score: {scope['hpc_relevance_score']}",
        f"Grounding score: {gaps['grounding_score']}",
        f"Ready for LLM remediation: {gaps['ready_for_llm']}",
        f"Runtime score: {scope.get('runtime_hpc_score', 0)}",
        f"Platform score: {scope.get('platform_hpc_score', 0)}",
        f"Timestamp hint: {inc['timestamp_hint'] or 'none'}",
        f"Total non-empty log lines: {inc['line_count']}",
        f"Observed hosts: {', '.join(item['value'] for item in topology['hosts'][:5]) or 'none'}",
        f"Observed ranks: {', '.join(item['value'] for item in topology['ranks'][:8]) or 'none'}",
        f"Observed interfaces: {', '.join(item['value'] for item in topology['network_interfaces'][:5]) or 'none'}",
        f"Mixed transcript detected: {structure['mixed_sources']}",
        "",
        "Top signals:",
    ]
    parts.extend(f"- {line}" for line in inc["top_signals"])
    high_priority_gaps = [gap["gap"] for gap in gaps["evidence_gaps"] if gap["priority"] == "high"]
    if high_priority_gaps:
        parts.extend(["", "High-priority evidence gaps:"])
        parts.extend(f"- {gap}" for gap in high_priority_gaps[:5])
    if templates["top_templates"]:
        parts.extend(["", "Top recurring templates:"])
        parts.extend(
            f"- {entry['count']}x {entry['template']}"
            for entry in templates["top_templates"][:5]
        )
    parts.extend(
        [
            "",
            "Redacted preview:",
            inc["redacted_preview"],
        ]
    )
    return "\n".join(parts)


def _find_matching_display_lines(text: str, patterns: list[re.Pattern], limit: int = 3) -> list[str]:
    matches = []
    for idx, line in enumerate(_non_empty_lines(redact_secrets(text)), start=1):
        if any(pattern.search(line) for pattern in patterns):
            matches.append(_display_line(idx, line))
            if len(matches) >= limit:
                break
    return matches


def _extract_missing_activate_path(text: str) -> str | None:
    match = _ACTIVATE_MISSING_RX.search(redact_secrets(text))
    return match.group(1) if match else None


def _summarize_local_rank_mapping(text: str) -> dict:
    pairs = []
    counts: Counter = Counter()
    lines = []
    for idx, line in enumerate(_non_empty_lines(redact_secrets(text)), start=1):
        match = _RANK_LOCAL_RANK_RX.search(line)
        if not match:
            continue
        rank = match.group(1)
        local_rank = match.group(2)
        pairs.append((rank, local_rank))
        counts[local_rank] += 1
        lines.append(_display_line(idx, line))

    duplicated_local_ranks = [
        {"local_rank": local_rank, "count": count}
        for local_rank, count in counts.items()
        if count > 1
    ]
    return {
        "pairs": pairs,
        "duplicated_local_ranks": duplicated_local_ranks,
        "lines": lines[:6],
    }


def _build_runtime_provisional_remediation(text: str, env_context: str = "") -> str:
    clean = redact_secrets(text)
    incident = extract_incident(clean)
    gaps = assess_evidence_gaps(clean, env_context)
    structure = analyze_log_structure(clean)
    scope = incident["scope_assessment"]

    activate_path = _extract_missing_activate_path(clean)
    missing_activate_lines = _find_matching_display_lines(
        clean,
        [re.compile(r"(?i)\bactivate:\s+No such file or directory\b"), re.compile(r"(?i)\bbin/activate:\s+No such file or directory\b")],
        limit=2,
    )
    torch_missing_lines = _find_matching_display_lines(
        clean,
        [re.compile(r"(?i)\bModuleNotFoundError:\s+No module named ['\"]torch['\"]")],
        limit=4,
    )
    timeout_lines = _find_matching_display_lines(
        clean,
        [re.compile(r"(?i)\bNCCL\b.*\bWARN\b.*\bTimeout\b"), re.compile(r"(?i)\bTimeout waiting for rank\b"), re.compile(r"(?i)\btimed out\b")],
        limit=4,
    )
    scheduler_exit_lines = _find_matching_display_lines(
        clean,
        [re.compile(r"(?i)\b(?:srun|slurmstepd):\s+error:.*Exited with exit code\b"), re.compile(r"(?i)\bExited with exit code\b")],
        limit=4,
    )
    nccl_plugin_error_lines = _find_matching_display_lines(
        clean,
        [re.compile(r"(?i)\bNET/Plugin: Plugin load returned\b.*\blibnccl-net\.so\b")],
        limit=2,
    )
    nccl_fallback_lines = _find_matching_display_lines(
        clean,
        [
            re.compile(r"(?i)\bUsing internal network plugin\b"),
            re.compile(r"(?i)\bUsing network IB\b"),
            re.compile(r"(?i)\bConnected all rings\b"),
        ],
        limit=4,
    )
    barrier_warning_lines = _find_matching_display_lines(
        clean,
        [re.compile(r"(?i)\busing GPU \d+ to perform barrier as devices used by this process are currently unknown\b")],
        limit=4,
    )
    local_rank_summary = _summarize_local_rank_mapping(clean)

    env_conflict = bool(
        env_context
        and re.search(r"(?i)\bno slurm\b", env_context)
        and re.search(r"(?i)\bslurm\b|srun|slurmd", clean)
    )
    env_bootstrap_issue = bool(missing_activate_lines or torch_missing_lines)
    mapping_issue = bool(barrier_warning_lines or local_rank_summary["duplicated_local_ranks"])

    status_line = (
        "Status: `provisional` because the upload is mixed or still under-grounded. "
        "This is a deterministic triage brief, not a hallucinated final fix plan."
        if structure["mixed_sources"] or not gaps["ready_for_llm"]
        else "Status: `grounded enough` for deeper synthesis."
    )

    lines = [
        "## Evidence-First Remediation",
        "",
        status_line,
        "",
        "### Primary blocker",
    ]

    if missing_activate_lines and torch_missing_lines:
        lines.extend(
            [
                "Broken Python environment bootstrap is the first hard failure.",
                "",
                "Evidence:",
                *[f"- {line}" for line in missing_activate_lines],
                *[f"- {line}" for line in torch_missing_lines[:2]],
                "",
                "Interpretation:",
                "The launch script tries to source a missing environment, then the Python interpreter that actually starts the job does not have `torch` available. "
                "That is a direct explanation for the rank exits and it appears before the NCCL noise.",
            ]
        )
    elif torch_missing_lines:
        lines.extend(
            [
                "The launched Python interpreter cannot import `torch`.",
                "",
                "Evidence:",
                *[f"- {line}" for line in torch_missing_lines[:3]],
                "",
                "Interpretation:",
                "Treat this as a broken runtime environment first, not as a fabric incident.",
            ]
        )
    elif timeout_lines:
        lines.extend(
            [
                "The current failing artifact points to a collective or communicator stall.",
                "",
                "Evidence:",
                *[f"- {line}" for line in timeout_lines[:3]],
                "",
                "Interpretation:",
                "This looks more like a distributed runtime coordination problem than a missing Python package. "
                "Use per-rank logs plus rank-to-device and fabric checks before changing application code.",
            ]
        )
    else:
        lines.extend(
            [
                "The first failing artifact is not yet specific enough for a single root-cause claim.",
                "",
                "Evidence:",
                *[f"- {line}" for line in incident["top_signals"][:3]],
            ]
        )

    if scheduler_exit_lines:
        lines.extend(
            [
                "",
                "### Immediate fallout",
                *[f"- {line}" for line in scheduler_exit_lines[:3]],
                "",
                "Those rank exits look like downstream consequences of the earlier environment failure, not an independent root cause.",
            ]
        )

    if nccl_plugin_error_lines and nccl_fallback_lines:
        lines.extend(
            [
                "",
                "### Not the blocker right now",
                "The `libnccl-net.so` warning is noisy, but this upload does not justify blaming it for the current failure.",
                "",
                "Evidence:",
                *[f"- {line}" for line in nccl_plugin_error_lines],
                *[f"- {line}" for line in nccl_fallback_lines[:3]],
                "",
                "Why this matters:",
                "NVIDIA documents that external NCCL plugins can fall back to native/internal communication. "
                "Your embedded artifact also proceeds into IB transport and ring connection, which weakens the case that the missing plugin is the immediate cause here.",
            ]
        )

    if barrier_warning_lines or local_rank_summary["duplicated_local_ranks"]:
        lines.extend(
            [
                "",
                "### Latent DDP / device-mapping risk",
                "The embedded comparison artifact shows a real distributed-training issue that can bite you after the environment is fixed.",
                "",
                "Evidence:",
            ]
        )
        if local_rank_summary["lines"]:
            lines.extend(f"- {line}" for line in local_rank_summary["lines"][:4])
        if barrier_warning_lines:
            lines.extend(f"- {line}" for line in barrier_warning_lines[:3])
        lines.extend(
            [
                "",
                "Interpretation:",
                "Multiple ranks appear to report `Local rank: 0`, and PyTorch warns that NCCL is choosing GPU 0 for barrier because device usage is unknown. "
                "That is not the cause of the current `torch` import failure, but it is a credible next failure mode once the job starts running.",
            ]
        )

    if structure["mixed_sources"]:
        artifact_names = [item["path"] for item in structure["embedded_artifacts"][:3] if item.get("path")]
        lines.extend(
            [
                "",
                "### Why the mixed upload matters",
                "This paste combines at least one failing artifact with an embedded shell transcript and another log view.",
            ]
        )
        if artifact_names:
            lines.append(f"Detected embedded artifact(s): `{', '.join(artifact_names)}`")
        lines.extend(
            [
                "",
                "That makes the evidence useful for human triage, but unsafe for a single monolithic fix synthesis because one slice fails immediately on environment setup while another slice progresses into NCCL communicator setup.",
            ]
        )

    if env_conflict:
        lines.extend(
            [
                "",
                "### Environment mismatch worth resolving",
                "Your environment textbox says `Scheduler: (no Slurm) / internal orchestration`, but the log clearly contains Slurm paths and `srun`/`slurmd` lines.",
                "Treat that mismatch as a hint that the upload is mixing runs or environments.",
            ]
        )

    lines.extend(["", "### Run next"])
    if env_bootstrap_issue:
        if activate_path:
            login_probe = f"ls -ld {activate_path}"
        else:
            login_probe = "ls -ld <expected-env>/bin/activate"
        lines.extend(
            [
                "Fix and validate the runtime environment before touching NCCL tuning.",
                "",
                "```bash",
                login_probe,
                "python -c \"import sys; print(sys.executable)\"",
                "python -c \"import torch; print(torch.__version__)\"",
                "srun -N1 -n1 bash -lc 'which python; python -c \"import sys; print(sys.executable)\"; python -c \"import torch; print(torch.__version__)\"'",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "Collect rank, GPU, and fabric state before changing libraries or timeouts.",
                "",
                "```bash",
                "env | egrep 'RANK|LOCAL_RANK|WORLD_SIZE|CUDA_VISIBLE_DEVICES|NCCL' || true",
                "nvidia-smi",
                "nvidia-smi topo -m",
                "ibstat || true",
                "ibv_devinfo || true",
                "```",
            ]
        )

    if mapping_issue:
        lines.extend(
            [
                "",
                "If the local-rank mapping is still ambiguous, bind each process to its GPU explicitly:",
                "",
                "```python",
                "local_rank = int(os.environ[\"LOCAL_RANK\"])",
                "torch.cuda.set_device(local_rank)",
                "dist.init_process_group(\"nccl\", device_id=local_rank)",
                "model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)",
                "dist.barrier(device_ids=[local_rank])",
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "### Primary references",
            f"- NVIDIA NCCL plugin behavior: {PRIMARY_SOURCE_LINKS['nccl_plugin_fallback']}",
            f"- PyTorch barrier device selection and `device_id`: {PRIMARY_SOURCE_LINKS['torch_distributed_barrier']}",
            f"- PyTorch DDP device mapping (`device_ids=[i]`, `init_process_group(device_id=i)`): {PRIMARY_SOURCE_LINKS['torch_ddp_device_mapping']}",
        ]
    )

    if scope.get("workflow_profile") != "runtime_ops":
        lines.extend(
            [
                "",
                f"Scope note: current profile is `{scope.get('workflow_profile', 'unknown')}`. The runtime guidance above is included because the pasted sample still contains strong GPU/NCCL/Slurm evidence.",
            ]
        )

    return "\n".join(lines)


def build_evidence_first_remediation(text: str, env_context: str = "") -> str:
    scope = assess_incident_scope(text, env_context)
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")

    if profile == "runtime_ops":
        return _build_runtime_provisional_remediation(text, env_context)

    reason = scope["explanation"]
    gaps = assess_evidence_gaps(text, env_context)
    lines = [
        "## Evidence-First Remediation",
        "",
        "Status: `provisional` because the current upload does not justify a final synthesized fix plan yet.",
        "",
        f"Current classification: `{profile}`",
        "",
        reason,
    ]
    if gaps["evidence_gaps"]:
        lines.extend(
            [
                "",
                "Top evidence gaps:",
                *[f"- {gap['gap']}" for gap in gaps["evidence_gaps"][:4]],
            ]
        )
    lines.extend(
        [
            "",
            "Next move: add the missing artifact boundaries and rerun remediation after the failure class is anchored.",
        ]
    )
    return "\n".join(lines)


# ============================
# Runbooks (deterministic, tag-driven)
# ============================

def build_runbooks(tags: list[str]) -> str:
    blocks = []

    blocks.append(
        "### Basic\n"
        "uptime\n"
        "hostname\n"
        "date\n"
        "df -h\n"
        "free -h || true\n"
        "dmesg | tail -200 || true\n"
    )

    if "GPU" in tags or "OOM" in tags:
        blocks.append(
            "### GPU / CUDA\n"
            "nvidia-smi\n"
            "nvidia-smi -q | head -200\n"
            "# If XID appears in dmesg, it’s often a driver/firmware/hardware signal.\n"
        )

    if "InfiniBand" in tags:
        blocks.append(
            "### InfiniBand\n"
            "ibstat || true\n"
            "ibv_devinfo || true\n"
            "ip a\n"
            "ip r\n"
        )

    if "NCCL" in tags:
        blocks.append(
            "### NCCL\n"
            "export NCCL_DEBUG=INFO\n"
            "export NCCL_DEBUG_SUBSYS=ALL\n"
            "export NCCL_ASYNC_ERROR_HANDLING=1\n"
            "export NCCL_SOCKET_IFNAME=bond0\n"
            "export NCCL_IB_DISABLE=0\n"
        )

    if "Network" in tags:
        blocks.append(
            "### Network\n"
            "ip a\n"
            "ip r\n"
            "ping -c 2 <peer_ip> || true\n"
            "ss -tulpn | head || true\n"
        )

    if "Disk/FS" in tags:
        blocks.append(
            "### Storage / FS\n"
            "mount | egrep 'nfs|vast' || true\n"
            "df -h\n"
            "ls -lah /mnt || true\n"
            "# If stale handle: consider remount or check server-side issues.\n"
        )

    if "Scheduler" in tags:
        blocks.append(
            "### Scheduler (Slurm/PBS/LSF)\n"
            "# Slurm examples:\n"
            "squeue -u $USER || true\n"
            "sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,Elapsed,NodeList || true\n"
            "scontrol show job $SLURM_JOB_ID || true\n"
        )

    if "Container" in tags:
        blocks.append(
            "### Container\n"
            "docker ps -a || true\n"
            "podman ps -a || true\n"
            "apptainer --version || true\n"
            "singularity --version || true\n"
        )

    if "Platform" in tags:
        blocks.append(
            "### Platform / Orchestrator\n"
            "id && whoami\n"
            "ps -ef | egrep 'sshd|determined|bash|python' || true\n"
            "ss -tulpn | egrep '22|3282' || true\n"
            "ls -ld /run /var/run /run/determined || true\n"
            "cat /etc/ssh/sshd_config || true\n"
        )

    return "\n\n".join(blocks).strip()


# ============================
# Inputs
# ============================

def load_file_to_text(file) -> str:
    if not file:
        return ""
    files = file if isinstance(file, list) else [file]
    loaded_parts = []

    for entry in files:
        path_value = getattr(entry, "name", entry)
        path = Path(str(path_value))
        data = path.read_bytes()
        try:
            content = data.decode("utf-8")
        except Exception:
            content = data.decode("latin-1", errors="replace")
        loaded_parts.append(f"===== FILE: {path.name} =====\n{content.strip()}")

    return "\n\n".join(part for part in loaded_parts if part.strip())


def ocr_image_to_text(image) -> str:
    if image is None:
        return ""
    if pytesseract is None:
        return "ERROR: pytesseract is not installed in this environment."
    try:
        # image is PIL.Image (because gr.Image(type="pil"))
        return (pytesseract.image_to_string(image) or "").strip()
    except Exception as e:
        return f"ERROR: OCR failed: {e}"


def build_incident_bundle(text: str, env_context: str) -> dict:
    incident = extract_incident(text)
    return {
        "incident": incident,
        "scope_assessment": incident["scope_assessment"],
        "environment_context": env_context.strip(),
        "log_structure": analyze_log_structure(text),
        "topology": build_incident_topology(text),
        "template_mining": mine_log_templates(text),
        "structured_signatures": parse_hpc_failure_artifacts(text),
        "runbooks": build_runbooks(incident["tags"]),
        "official_guidance": get_official_guidance(incident["tags"]),
        "local_memory": find_similar_incidents(text, limit=3),
        "collection_manifest": build_collection_manifest(text, env_context),
    }


def suggest_mcp_companions(text: str) -> dict:
    incident = extract_incident(text)
    tags = incident["tags"]
    scope = incident["scope_assessment"]
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")
    topology = build_incident_topology(text)
    signatures = parse_hpc_failure_artifacts(text)
    gaps = assess_evidence_gaps(text)
    browser_ready = _supports_browser_investigation(profile, topology, gaps["grounding_score"], signatures)
    telemetry_ready = _supports_telemetry_investigation(topology, gaps["grounding_score"])

    if profile == "access_ops":
        return {
            "incident_id": incident["id"],
            "tags": tags,
            "scope_assessment": scope,
            "recommended_servers": [
                {
                    "server": "filesystem",
                    "priority": "high",
                    "why": "Open the actual SSH config, known_hosts entries, and any cluster-specific bootstrap scripts referenced in the trace.",
                    "example_tasks": [
                        "Inspect ~/.ssh/config and ProxyJump aliases",
                        "Open known_hosts entries for the login and jump hosts",
                        "Read any local wrapper scripts used to reach the cluster",
                    ],
                },
                {
                    "server": "shell",
                    "priority": "high",
                    "why": "Run deterministic access checks before escalating into GPU/fabric workflows.",
                    "example_tasks": [
                        "Run ssh -G <alias> to inspect the resolved SSH configuration",
                        "Retry the jump/login path with verbose flags and the intended username",
                        "Test direct connectivity to the login host on port 22",
                    ],
                },
            ],
        }

    if profile == "generic_ops":
        return {
            "incident_id": incident["id"],
            "tags": tags,
            "scope_assessment": scope,
            "recommended_servers": [
                {
                    "server": "filesystem",
                    "priority": "high",
                    "why": "Open adjacent logs, configs, and launch artifacts to give the current sample enough surrounding context.",
                    "example_tasks": [
                        "Read neighboring .log/.out/.err files from the same run",
                        "Inspect launch scripts or config files referenced by the failure",
                        "Compare pre-failure and post-failure log slices",
                    ],
                },
                {
                    "server": "shell",
                    "priority": "high",
                    "why": "Run broad deterministic system checks before assuming a specialized GPU/fabric failure mode.",
                    "example_tasks": [
                        "Check process, port, disk, and memory state",
                        "Collect recent dmesg or kernel warnings",
                        "Confirm the exact non-zero exit path if one is present",
                    ],
                },
            ],
        }

    if profile == "platform_ops":
        return {
            "incident_id": incident["id"],
            "tags": tags,
            "scope_assessment": scope,
            "recommended_servers": [
                {
                    "server": "filesystem",
                    "priority": "high",
                    "why": "Open platform startup hooks, sshd config, allocation artifacts, and adjacent task logs inside the containerized shell environment.",
                    "example_tasks": [
                        "Inspect /run/determined assets and startup-hook.sh",
                        "Open sshd_config and runtime directories such as /run and /var/run",
                        "Read agent/master task-event exports if they are available locally",
                    ],
                },
                {
                    "server": "shell",
                    "priority": "high",
                    "why": "Run deterministic container and service checks before assuming a training-runtime failure.",
                    "example_tasks": [
                        "Verify the container user, process tree, and listening ports",
                        "Check whether /run/sshd.pid and login-record paths are compatible with the current user",
                        "Capture platform and GPU visibility environment variables",
                    ],
                },
                {
                    "server": "playwright",
                    "priority": "medium",
                    "why": "Inspect the Determined or platform UI for allocation history, restore events, kill reasons, and resource-pool context.",
                    "example_tasks": [
                        "Open the task or shell page and inspect lifecycle events",
                        "Capture the allocation timeline around restores and termination",
                        "Screenshot the resource-pool, workspace, or task-state panels",
                    ],
                },
                {
                    "server": "browser/docs",
                    "priority": "medium",
                    "why": "Cross-check commands/shells and resource-pool behavior against official platform docs before calling the issue a generic container failure.",
                    "example_tasks": [
                        "Read Determined commands and shells docs",
                        "Check resource-pool configuration guidance",
                        "Verify whether shell services are expected to run sshd in a constrained container user context",
                    ],
                },
            ],
        }

    servers = [
        {
            "server": "filesystem",
            "priority": "high",
            "why": "Read sibling stdout/stderr files, scheduler outputs, dmesg exports, NCCL configs, and node-specific artifacts near the failing logs.",
            "example_tasks": [
                "Open neighboring rank logs from the same job",
                "Inspect /etc/nccl.conf or user-level NCCL config",
                "Read saved dmesg or scheduler output files",
            ],
        },
        {
            "server": "shell",
            "priority": "high",
            "why": "Run the next deterministic diagnostic commands produced by Log Alchemist against the affected node or cluster environment.",
            "example_tasks": [
                "Run nvidia-smi, ibstat, ibv_devinfo, and network checks",
                "Query scheduler state with sacct/scontrol when applicable",
                "Collect fresh dmesg and interface status",
            ],
        },
    ]

    if browser_ready and any(tag in tags for tag in ["GPU", "OOM", "NCCL", "Scheduler", "Disk/FS", "Network", "InfiniBand"]):
        servers.append(
            {
                "server": "playwright",
                "priority": "medium",
                "why": "Inspect Grafana, scheduler portals, Kibana, storage dashboards, or internal cluster UIs and capture browser-based evidence with deterministic automation.",
                "example_tasks": [
                    "Open the Grafana time window around the failure and capture GPU/fabric anomalies",
                    "Inspect scheduler or job portal pages for exit codes, node lists, and failed steps",
                    "Open storage or network dashboards and screenshot relevant incident panels",
                ],
            }
        )

    if any(tag in tags for tag in ["NCCL", "GPU", "InfiniBand", "Scheduler"]):
        servers.append(
            {
                "server": "browser/docs",
                "priority": "medium",
                "why": "Cross-check the observed failure against official NCCL, PyTorch Elastic, or Slurm guidance rather than relying on generic guesses.",
                "example_tasks": [
                    "Look up NCCL environment variables and transport guidance",
                    "Check PyTorch Elastic root-cause summary behavior",
                    "Read Slurm troubleshooting docs for scheduler-specific failures",
                ],
            }
        )

    if telemetry_ready and any(tag in tags for tag in ["NCCL", "Network", "InfiniBand"]):
        servers.append(
            {
                "server": "metrics/telemetry",
                "priority": "medium",
                "why": "Correlate log failures with NIC, GPU, or scheduler-level telemetry if your environment exposes those through MCP.",
                "example_tasks": [
                    "Check GPU/Xid event spikes around the failure window",
                    "Check fabric or link-health counters by node",
                    "Compare failure timing with cluster-wide anomalies",
                ],
            }
        )

    return {
        "incident_id": incident["id"],
        "tags": tags,
        "scope_assessment": scope,
        "recommended_servers": servers,
    }


def assess_evidence_gaps(text: str, env_context: str = "") -> dict:
    incident = extract_incident(text)
    topology = build_incident_topology(text)
    templates = mine_log_templates(text)
    signatures = parse_hpc_failure_artifacts(text)
    structure = analyze_log_structure(text)
    scope = assess_incident_scope(text, env_context)
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")

    gaps = []
    if profile == "access_ops":
        gaps.append(
            {
                "gap": "Low HPC relevance for this log sample",
                "priority": "high",
                "why": scope["explanation"],
                "how_to_collect": "Provide job stdout/stderr, scheduler outputs, TorchElastic error files, or node-local runtime logs before invoking GPU/fabric workflows.",
            }
        )
    elif profile == "generic_ops":
        gaps.append(
            {
                "gap": "Failure class is still ambiguous",
                "priority": "high",
                "why": "The current sample does not yet anchor the incident to a clear runtime, platform, or access workflow.",
                "how_to_collect": "Add adjacent logs, process/exit-code context, and any launch/config artifacts so the failure class can be pinned down.",
            }
        )
    if structure["mixed_sources"]:
        gaps.append(
            {
                "gap": "Mixed transcript and embedded artifacts detected",
                "priority": "high",
                "why": "Interactive shell prompts and embedded file views were merged into one upload, which can scramble causality and line ranking.",
                "how_to_collect": "Upload the failing job log, scheduler output, and any comparison logs as separate artifacts instead of one pasted terminal transcript.",
            }
        )
    if incident["line_count"] < 6 and not topology["hosts"] and not topology["job_ids"]:
        gaps.append(
            {
                "gap": "Sample is too small for agentic routing",
                "priority": "high",
                "why": "A tiny log slice without host or job context is enough for tagging, but not enough for browser missions, telemetry pivots, or LLM remediation.",
                "how_to_collect": "Add surrounding lines, the failing rank log, and at least one host/job/context anchor before requesting an agentic plan.",
            }
        )
    if not topology["hosts"]:
        gaps.append(
            {
                "gap": "No host names detected",
                "priority": "high",
                "why": "Host-level correlation is crucial for distributed failures.",
                "how_to_collect": "Gather per-node logs or include hostname-prefixed outputs.",
            }
        )
    if "NCCL" in incident["tags"] and not topology["ranks"]:
        gaps.append(
            {
                "gap": "No rank identifiers detected",
                "priority": "high",
                "why": "NCCL failures are often localized to one or a few ranks.",
                "how_to_collect": "Include torchrun / elastic summaries or per-rank stdout/stderr files.",
            }
        )
    if "GPU" in incident["tags"] and not topology["gpus"]:
        gaps.append(
            {
                "gap": "No GPU identifiers detected",
                "priority": "medium",
                "why": "GPU-local failures are harder to isolate without device ids.",
                "how_to_collect": "Add nvidia-smi, rank-local logs, or CUDA device annotations.",
            }
        )
    if "Scheduler" in incident["tags"] and not topology["job_ids"]:
        gaps.append(
            {
                "gap": "No scheduler job id detected",
                "priority": "medium",
                "why": "Job metadata is needed for node list, exit code, and history.",
                "how_to_collect": "Include job environment or scheduler command outputs.",
            }
        )
    if profile == "platform_ops":
        platform = signatures["platform"]
        if not platform["allocation_ids"]:
            gaps.append(
                {
                    "gap": "No allocation or task identifiers detected",
                    "priority": "medium",
                    "why": "Platform incidents are easier to trace when task and allocation ids are available end-to-end.",
                    "how_to_collect": "Include the orchestration event history or the original task launch metadata.",
                }
            )
        if 137 in platform["generic_exit_codes"] and not platform["user_requested_kill"]:
            gaps.append(
                {
                    "gap": "Exit code 137 lacks a confirmed cause",
                    "priority": "high",
                    "why": "SIGKILL can mean manual termination, orchestrator policy, or OOM; the current sample does not disambiguate it.",
                    "how_to_collect": "Gather master/agent events, container runtime events, and kernel OOM evidence for the same time window.",
                }
            )
        if platform["ssh_service_warnings"] and not platform["service_ready"]:
            gaps.append(
                {
                    "gap": "Interactive service warnings without confirmed readiness",
                    "priority": "high",
                    "why": "The shell service may have failed before becoming usable.",
                    "how_to_collect": "Collect the shell service readiness events, port listeners, and any client connection attempts.",
                }
            )
        if platform["user_requested_kill"]:
            gaps.append(
                {
                    "gap": "Kill request origin not yet verified",
                    "priority": "medium",
                    "why": "User-requested kills are often intentional, so they should not be triaged like spontaneous platform failures.",
                    "how_to_collect": "Check the platform event history or UI to confirm who initiated the termination.",
                }
            )
    if templates["unique_templates"] <= 3 and incident["line_count"] > 50:
        gaps.append(
            {
                "gap": "Low template diversity",
                "priority": "medium",
                "why": "The logs may only contain a noisy slice and be missing surrounding context.",
                "how_to_collect": "Include more pre-failure and post-failure context or sibling logs.",
            }
        )

    if profile == "access_ops":
        score = 0.2
    elif profile == "generic_ops":
        score = 0.25
    elif profile == "platform_ops":
        score = 0.45
    else:
        score = 0.35
    if topology["hosts"]:
        score += 0.15
    if topology["ranks"] or "NCCL" not in incident["tags"]:
        score += 0.15
    if topology["job_ids"] or "Scheduler" not in incident["tags"]:
        score += 0.05
    if topology["gpus"] or "GPU" not in incident["tags"]:
        score += 0.05
    if any(section.get("detected") for section in signatures.values()):
        score += 0.1
    if profile == "platform_ops" and signatures["platform"]["service_ready"]:
        score += 0.05
    if profile == "platform_ops" and signatures["platform"]["user_requested_kill"]:
        score += 0.05
    if get_official_guidance(incident["tags"]):
        score += 0.05
    if not any(gap["priority"] == "high" for gap in gaps):
        score += 0.1
    if incident["line_count"] < 5:
        score -= 0.1
    if incident["line_count"] < 8:
        score -= 0.05
    if structure["mixed_sources"]:
        score -= 0.15
    if profile == "access_ops":
        score = min(score, 0.35)

    grounding_score = round(max(0.0, min(0.95, score)), 2)
    confidence_label = (
        "high" if grounding_score >= 0.75 else
        "medium" if grounding_score >= 0.55 else
        "low"
    )

    if profile == "access_ops":
        expected_artifacts = [
            {
                "artifact": "Resolved SSH config, usernames, and jump-host chain",
                "why": "This sample looks like a cluster access/bootstrap issue.",
            },
            {
                "artifact": "Actual runtime logs from inside the cluster",
                "why": "HPC remediation should start from job/runtime evidence, not from the login bootstrap trace.",
            },
        ]
    elif profile == "platform_ops":
        expected_artifacts = [
            {
                "artifact": "Platform allocation/task timeline",
                "why": "Restores, service readiness, and kills should be interpreted using orchestrator events rather than only raw container logs.",
            },
            {
                "artifact": "Startup hook and container-user configuration",
                "why": "The non-root sshd warnings point at how the interactive shell is packaged and launched.",
            },
        ]
    elif profile == "generic_ops":
        expected_artifacts = [
            {
                "artifact": "Adjacent logs, launch configs, and process metadata",
                "why": "The incident class is not clear enough yet to jump into a specialized workflow.",
            }
        ]
    else:
        expected_artifacts = _expected_artifacts_for_tags(incident["tags"])

    return {
        "incident_id": incident["id"],
        "tags": incident["tags"],
        "expected_artifacts": expected_artifacts,
        "evidence_gaps": gaps,
        "log_structure": structure,
        "ready_for_llm": profile in {"runtime_ops", "platform_ops"} and not any(gap["priority"] == "high" for gap in gaps),
        "grounding_score": grounding_score,
        "confidence_label": confidence_label,
        "scope_assessment": scope,
    }


def build_agentic_response_plan(text: str, env_context: str = "") -> dict:
    bundle = build_incident_bundle(text, env_context)
    companions = suggest_mcp_companions(text)["recommended_servers"]
    gaps = assess_evidence_gaps(text, env_context)
    manifest = build_collection_manifest(text, env_context)
    scope = bundle["incident"]["scope_assessment"]
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")

    if profile == "access_ops":
        return {
            "incident_id": bundle["incident"]["id"],
            "title": "Log Alchemist — Agentic Boundary Check",
            "tags": bundle["incident"]["tags"],
            "scope_assessment": scope,
            "companion_servers": companions,
            "evidence_gaps": gaps,
            "collection_manifest": manifest,
            "agents": [
                {
                    "agent": "Access Path Analyst",
                    "mission": "Verify SSH configuration, usernames, ProxyJump behavior, and authentication sequence.",
                },
                {
                    "agent": "Evidence Collector Agent",
                    "mission": "Collect the actual job/runtime logs from inside the cluster before escalating into HPC runtime workflows.",
                },
            ],
            "phases": [
                {
                    "phase": "1. Scope boundary check",
                    "owner": "Log Alchemist MCP",
                    "actions": [
                        "Confirm that the current sample is an access/bootstrap trace rather than a runtime training failure",
                        "Review the scope assessment and suppress GPU/fabric assumptions",
                    ],
                    "stop_when": "You know whether this is an SSH/bootstrap issue or a real runtime incident.",
                },
                {
                    "phase": "2. Access-path diagnostics",
                    "owner": "Filesystem + Shell MCP",
                    "actions": [
                        "Inspect SSH config and known_hosts",
                        "Run ssh -G and ssh -vvv against the intended cluster alias",
                        "Verify the username, key, and jump-host chain",
                    ],
                    "stop_when": "The login path succeeds or the exact authentication failure is identified.",
                },
                {
                    "phase": "3. Runtime evidence acquisition",
                    "owner": "Operator",
                    "actions": [
                        "Collect scheduler outputs, job stdout/stderr, or runtime logs from inside the cluster",
                        "Re-run Log Alchemist on those logs before using HPC-specific remediation flows",
                    ],
                    "stop_when": "You have real runtime evidence for a possible HPC incident.",
                },
            ],
        }

    if profile == "generic_ops":
        return {
            "incident_id": bundle["incident"]["id"],
            "title": "Log Alchemist — Agentic Generic Triage",
            "tags": bundle["incident"]["tags"],
            "scope_assessment": scope,
            "companion_servers": companions,
            "evidence_gaps": gaps,
            "collection_manifest": manifest,
            "agents": [
                {
                    "agent": "Log Analyst Agent",
                    "mission": "Classify the failure and keep the investigation grounded in the uploaded evidence.",
                },
                {
                    "agent": "Evidence Collector Agent",
                    "mission": "Collect adjacent logs, process state, and launch artifacts until the failure class becomes clear.",
                },
            ],
            "phases": [
                {
                    "phase": "1. Failure-class triage",
                    "owner": "Log Alchemist MCP",
                    "actions": [
                        "Review tags, top signals, and the scope assessment",
                        "Avoid forcing the sample into an SSH or GPU/fabric explanation too early",
                    ],
                    "stop_when": "You know what broad class of issue this log belongs to.",
                },
                {
                    "phase": "2. Context expansion",
                    "owner": "Filesystem + Shell MCP",
                    "actions": [
                        "Collect adjacent logs and configs",
                        "Gather process, port, disk, memory, and kernel context",
                        "Re-run Log Alchemist with the expanded evidence set",
                    ],
                    "stop_when": "The incident is anchored to a specific workflow profile.",
                },
            ],
        }

    if profile == "platform_ops":
        platform_agents = [
            {
                "agent": "Platform Log Analyst",
                "mission": "Interpret allocation, task, resource-pool, and container lifecycle signals without overfitting to runtime NCCL failures.",
            },
            {
                "agent": "Container Service Investigator",
                "mission": "Validate the non-root shell service, sshd behavior, runtime directories, and exit-code semantics inside the container.",
            },
        ]
        platform_phases = [
            {
                "phase": "1. Platform triage",
                "owner": "Log Alchemist MCP",
                "actions": [
                    "Build the incident bundle and inspect structured_signatures.platform",
                    "Decide whether the key issue is startup, service readiness, or intentional termination",
                    "Separate secondary sshd warnings from true platform failure indicators",
                ],
                "stop_when": "You have a clear platform-oriented hypothesis with line-referenced evidence.",
            },
            {
                "phase": "2. Container and service diagnostics",
                "owner": "Filesystem + Shell MCP",
                "actions": [
                    "Inspect startup hooks, sshd config, runtime directories, and container user state",
                    "Verify whether the service actually became reachable despite warnings",
                    "Check whether exit code 137 was user-driven, policy-driven, or resource-driven",
                ],
                "stop_when": "The shell-service behavior and termination path are explained.",
            },
        ]
        if manifest["playwright"]:
            platform_agents.append(
                {
                    "agent": "Playwright Platform Scout",
                    "mission": "Inspect the platform UI for allocation history, restore events, kill reasons, and task-state transitions.",
                }
            )
            platform_phases.append(
                {
                    "phase": "3. Platform timeline correlation",
                    "owner": "Playwright MCP",
                    "actions": [
                        "Inspect the platform UI for restores, task transitions, and kill initiator",
                        "Capture screenshots of task, allocation, and resource-pool state",
                        "Record anything that confirms or contradicts the log-derived hypothesis",
                    ],
                    "stop_when": "The orchestrator timeline supports the final interpretation.",
                }
            )
        if gaps["ready_for_llm"]:
            platform_agents.append(
                {
                    "agent": "Incident Commander Agent",
                    "mission": "Turn platform evidence into an actionable handoff and operator-safe next steps.",
                }
            )
            platform_phases.append(
                {
                    "phase": f"{len(platform_phases) + 1}. Resolution and handoff",
                    "owner": "LLM + operator",
                    "actions": [
                        "Call suggest_fix once the platform evidence is strong enough",
                        "Document whether the issue was cosmetic, configuration-related, or operator-driven",
                        "Save the resolved case into local memory",
                    ],
                    "stop_when": "A human can explain and act on the platform incident with confidence.",
                }
            )
        else:
            platform_phases.append(
                {
                    "phase": f"{len(platform_phases) + 1}. Grounding checkpoint",
                    "owner": "Operator",
                    "actions": [
                        "Do not call suggest_fix yet",
                        "Close the remaining high-priority evidence gaps first",
                        "Only request final remediation after the platform evidence is grounded",
                    ],
                    "stop_when": "The incident is grounded enough for a final fix recommendation.",
                }
            )
        return {
            "incident_id": bundle["incident"]["id"],
            "title": "Log Alchemist — Agentic Platform Response",
            "tags": bundle["incident"]["tags"],
            "scope_assessment": scope,
            "companion_servers": companions,
            "evidence_gaps": gaps,
            "collection_manifest": manifest,
            "agents": platform_agents,
            "phases": platform_phases,
        }

    phases = [
        {
            "phase": "1. Grounded triage",
            "owner": "Log Alchemist MCP",
            "actions": [
                "Call build_incident_bundle",
                "Review topology, template mining, structured signatures, memory, and official guidance",
                "Identify top signals with line references",
            ],
            "stop_when": "You have a ranked problem framing and subsystem tags.",
        },
        {
            "phase": "2. Evidence gap closure",
            "owner": "Filesystem + Shell MCP",
            "actions": [
                "Collect missing artifacts listed in evidence gaps",
                "Execute the filesystem and shell tasks from the collection manifest",
                "Add any newly gathered error files, scheduler outputs, or per-rank logs back into the session",
            ],
            "stop_when": "High-priority evidence gaps are closed.",
        },
    ]
    if manifest["playwright"]:
        phases.append(
            {
                "phase": f"{len(phases) + 1}. Surface correlation",
                "owner": "Playwright MCP",
                "actions": [
                    "Inspect the recommended surfaces from the collection manifest",
                    "Capture screenshots or extracted browser evidence for impacted hosts/jobs/interfaces",
                    "Record anomalies that align with the log timeline",
                ],
                "stop_when": "Dashboard evidence either supports or refutes the log-based hypothesis.",
            }
        )
    if gaps["ready_for_llm"]:
        phases.append(
            {
                "phase": f"{len(phases) + 1}. Remediation and handoff",
                "owner": "LLM + operator",
                "actions": [
                    "Call suggest_fix after evidence collection",
                    "Prepare escalation handoff using the MCP prompt",
                    "Save the incident into local memory after resolution",
                ],
                "stop_when": "A human can execute the next safe steps with confidence.",
            }
        )
    else:
        phases.append(
            {
                "phase": f"{len(phases) + 1}. Grounding checkpoint",
                "owner": "Operator",
                "actions": [
                    "Do not call suggest_fix yet",
                    "Close the remaining high-priority evidence gaps first",
                    "Re-run the plan after adding the missing artifacts",
                ],
                "stop_when": "The incident is grounded enough for a final remediation pass.",
            }
        )

    agents = [
        {
            "agent": "Log Analyst Agent",
            "mission": "Use Log Alchemist tools to build the incident bundle and keep the investigation grounded.",
        },
        {
            "agent": "Evidence Collector Agent",
            "mission": "Use filesystem and shell MCP to fetch missing artifacts and run deterministic diagnostics.",
        },
    ]
    if manifest["playwright"]:
        agents.append(
            {
                "agent": "Playwright Scout Agent",
                "mission": "Use Playwright MCP to inspect dashboards and capture browser evidence around the failure window.",
            }
        )
    if gaps["ready_for_llm"]:
        agents.append(
            {
                "agent": "Incident Commander Agent",
                "mission": "Synthesize findings, produce the handoff, and coordinate the final answer.",
            }
        )

    return {
        "incident_id": bundle["incident"]["id"],
        "title": "Log Alchemist — Agentic Incident Response",
        "tags": bundle["incident"]["tags"],
        "scope_assessment": scope,
        "companion_servers": companions,
        "evidence_gaps": gaps,
        "collection_manifest": manifest,
        "agents": agents,
        "phases": phases,
    }


def generate_playwright_mission(text: str, env_context: str = "", surface: str = "Grafana dashboard") -> str:
    bundle = build_incident_bundle(text, env_context)
    scope = bundle["incident"]["scope_assessment"]
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")
    gaps = assess_evidence_gaps(text, env_context)
    if profile == "access_ops":
        return "\n".join(
            [
                f"# Playwright Mission Skipped: {surface}",
                "",
                "Log Alchemist did not generate a browser mission for this sample.",
                f"Reason: {scope['explanation']}",
                "",
                "Next step:",
                "- Fix the access/bootstrap issue first, or provide actual scheduler/job/runtime logs from inside the cluster.",
            ]
        )
    if profile == "generic_ops":
        return "\n".join(
            [
                f"# Playwright Mission Skipped: {surface}",
                "",
                "Log Alchemist did not generate a browser mission for this sample.",
                "Reason: the incident class is still too ambiguous to justify a browser-first workflow.",
                "",
                "Next step:",
                "- Expand the evidence set with adjacent logs and local diagnostics, then retry.",
            ]
        )

    topology = bundle["topology"]
    signatures = bundle["structured_signatures"]
    if not _supports_browser_investigation(profile, topology, gaps["grounding_score"], signatures):
        return "\n".join(
            [
                f"# Playwright Mission Skipped: {surface}",
                "",
                "Log Alchemist did not generate a browser mission for this sample.",
                "Reason: there are not enough concrete browser pivots yet (hosts, jobs, interfaces, allocations, or sufficient grounding).",
                "",
                "Next step:",
                "- Close the high-priority evidence gaps first, then retry Playwright generation.",
            ]
        )
    impacted_hosts = ", ".join(item["value"] for item in topology["hosts"][:5]) or "unknown hosts"
    impacted_jobs = ", ".join(item["value"] for item in topology["job_ids"][:5]) or "unknown jobs"
    impacted_ifaces = ", ".join(item["value"] for item in topology["network_interfaces"][:5]) or "unknown interfaces"

    if profile == "platform_ops":
        platform = bundle["structured_signatures"]["platform"]
        impacted_allocations = ", ".join(platform["allocation_ids"][:5]) or "unknown allocations"
        impacted_pools = ", ".join(platform["resource_pools"][:5]) or "unknown resource pools"
        return "\n".join(
            [
                f"# Playwright Mission: {surface}",
                "",
                "You are the Playwright Scout Agent for Log Alchemist — Agentic Platform Response.",
                "",
                "Mission:",
                f"- Inspect platform-visible evidence for allocations: {impacted_allocations}",
                f"- Check resource pools or workspaces related to: {impacted_pools}",
                f"- Correlate the lifecycle with hosts or containers tied to: {impacted_hosts}",
                "",
                "Playwright workflow:",
                "1. Open the Determined or platform UI page for the affected task, shell, or allocation.",
                "2. Inspect lifecycle events around startup, restore, and termination timestamps.",
                "3. Capture the task state, resource pool, workspace, and any kill initiator or event-reason fields.",
                "4. Screenshot service-readiness or connection details if the platform shows them.",
                "5. Summarize whether the UI confirms a harmless non-root shell warning, a configuration issue, or an intentional kill.",
                "",
                "Output requirements:",
                "- Include exact page or panel names.",
                "- Include task, allocation, resource-pool, or workspace identifiers.",
                "- Call out whether the UI shows a user-requested kill or restore event.",
                "- Note anything that sends the investigation back to filesystem or shell MCP.",
                "",
                "Relevant incident signals:",
                *[f"- {line}" for line in bundle["incident"]["top_signals"][:5]],
            ]
        )

    return "\n".join(
        [
            f"# Playwright Mission: {surface}",
            "",
            "You are the Playwright Scout Agent for Log Alchemist — Agentic Incident Response.",
            "",
            "Mission:",
            f"- Investigate browser-accessible evidence related to hosts: {impacted_hosts}",
            f"- Check any dashboards, portals, or UIs relevant to jobs: {impacted_jobs}",
            f"- Pay special attention to interfaces and fabric indicators involving: {impacted_ifaces}",
            "",
            "Playwright workflow:",
            "1. Open the relevant dashboard or portal.",
            "2. Set the time range around the failure timestamps from the incident bundle.",
            "3. Filter to the impacted hosts, jobs, GPUs, or interfaces where possible.",
            "4. Capture screenshots of the key panels or pages.",
            "5. Extract any visible error badges, status chips, counters, exit codes, or anomaly markers.",
            "6. Summarize what the UI evidence confirms, contradicts, or leaves unresolved.",
            "",
            "Output requirements:",
            "- Include exact panel/page names.",
            "- Include any extracted values, statuses, or labels.",
            "- Note anything that supports the log-derived root cause hypothesis.",
            "- Note anything surprising that should send the investigation back to filesystem or shell MCP.",
            "",
            "Relevant incident signals:",
            *[f"- {line}" for line in bundle["incident"]["top_signals"][:5]],
        ]
    )


def generate_mcp_investigation_prompt(text: str, env_context: str, objective: str) -> str:
    bundle = build_incident_bundle(text, env_context)
    companions = suggest_mcp_companions(text)["recommended_servers"]
    guidance = bundle["official_guidance"]
    scope = bundle["incident"]["scope_assessment"]
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")

    summary_lines = [
        f"Incident ID: {bundle['incident']['id']}",
        f"Tags: {', '.join(bundle['incident']['tags'])}",
        f"Top signals: {' | '.join(bundle['incident']['top_signals'][:4]) or 'none'}",
    ]

    prompt_lines = [f"# MCP Workflow: {objective}", ""]
    if profile == "runtime_ops":
        prompt_lines.extend(
            [
                "You are investigating an HPC incident using Log Alchemist MCP plus any available companion MCP servers.",
                "",
                "Use Log Alchemist first:",
                "1. Call `build_incident_bundle` with the current logs and environment context.",
                "2. Inspect `incident`, `topology`, `template_mining`, `structured_signatures`, `official_guidance`, and `local_memory`.",
                "3. For each detected tag, read the matching playbook resource such as `playbook://tag/NCCL`.",
                "4. If the incident seems recurrent, call `find_similar_incidents`.",
                "5. Call `assess_evidence_gaps` and close the high-priority gaps before trusting final conclusions.",
                "6. If browser-visible systems are relevant, call `generate_playwright_mission` and give that mission to a Playwright MCP agent.",
                "7. If you need a human-readable fix plan, call `suggest_fix` last, after evidence collection.",
                "",
                "If companion MCP servers are available, use them in this order:",
            ]
        )
    elif profile == "platform_ops":
        prompt_lines.extend(
            [
                "You are investigating an HPC platform or orchestrator incident using Log Alchemist MCP plus any available companion MCP servers.",
                "",
                "Use Log Alchemist first:",
                "1. Call `build_incident_bundle` and inspect `scope_assessment` plus `structured_signatures.platform`.",
                "2. Identify allocation ids, task types, resource pools, service-readiness lines, and termination semantics.",
                "3. Treat sshd pid/login-record warnings as secondary until the evidence shows they blocked readiness or access.",
                "4. Use filesystem and shell MCP to inspect startup hooks, container user state, sshd config, and runtime directories.",
                "5. If a platform UI exists, call `generate_playwright_mission` and inspect task/allocation history there.",
                "6. Call `suggest_fix` only after the platform lifecycle evidence is assembled.",
                "",
                "If companion MCP servers are available, use them in this order:",
            ]
        )
    elif profile == "access_ops":
        prompt_lines.extend(
            [
                "You are investigating a sample that currently looks like an access/bootstrap issue rather than an HPC runtime incident.",
                f"Reason: {scope['explanation']}",
                "",
                "Use Log Alchemist first:",
                "1. Call `build_incident_bundle` and inspect `scope_assessment`.",
                "2. Do not trigger GPU, NCCL, or Playwright workflows yet.",
                "3. Use filesystem and shell MCP to inspect SSH config, usernames, keys, and ProxyJump behavior.",
                "4. Collect the actual job/scheduler/runtime logs from inside the cluster.",
                "5. Re-run Log Alchemist on those runtime logs before asking for HPC remediation.",
                "",
                "If companion MCP servers are available, use them in this order:",
            ]
        )
    else:
        prompt_lines.extend(
            [
                "You are investigating a sample whose failure class is still ambiguous.",
                f"Reason: {scope['explanation']}",
                "",
                "Use Log Alchemist first:",
                "1. Call `build_incident_bundle` and inspect `scope_assessment` plus `top_signals`.",
                "2. Do not force the sample into an SSH or GPU/fabric explanation yet.",
                "3. Use filesystem and shell MCP to collect adjacent logs, configs, and basic process/system context.",
                "4. Re-run Log Alchemist on the expanded evidence set before asking for remediation.",
                "",
                "If companion MCP servers are available, use them in this order:",
            ]
        )
    for item in companions:
        prompt_lines.append(f"- `{item['server']}`: {item['why']}")

    if guidance:
        prompt_lines.extend(["", "Official guidance to consult when useful:"])
        prompt_lines.extend(f"- {item['title']}: {item['url']}" for item in guidance)

    answer_specificity = (
        "- Keep the answer specific to HPC/distributed training failure modes."
        if profile == "runtime_ops"
        else "- Keep the answer specific to the actual incident class and avoid forcing a GPU/fabric explanation."
    )
    prompt_lines.extend(
        [
            "",
            "Output requirements:",
            "- Cite evidence with line references from Log Alchemist outputs.",
            "- Distinguish current evidence from assumptions.",
            "- Give immediate safe actions first, then deeper diagnostics.",
            answer_specificity,
            "",
            "Current incident summary:",
        ]
    )
    prompt_lines.extend(f"- {line}" for line in summary_lines)
    return "\n".join(prompt_lines)


# ============================
# LLM remediation (chat models)
# ============================

def suggest_fix(logs: str, env_context: str, model: str) -> str:
    red = redact_secrets(logs)
    inc = extract_incident(red)
    scope = inc["scope_assessment"]
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")
    gaps = assess_evidence_gaps(red, env_context)
    provisional = build_evidence_first_remediation(red, env_context)
    if profile == "access_ops":
        return provisional
    if profile == "generic_ops":
        return provisional
    if not gaps["ready_for_llm"]:
        return provisional
    token = os.environ.get("HF_TOKEN")
    if not token:
        return (
            provisional
            + "\n\n## Remote Synthesis Unavailable\n\n"
            + "HF_TOKEN is not set in this app process, so the deterministic remediation above is the best available output right now."
        )
    evidence = build_llm_evidence(red)
    runbooks = build_runbooks(inc["tags"])

    if profile == "platform_ops":
        prompt = f"""
You are a Tier-3 HPC platform SRE specializing in:

- cluster orchestrators and resource pools
- containerized interactive shell services
- GPU-backed task lifecycle debugging
- Determined / HPE MLDE style task, allocation, and restore workflows

Environment:
{env_context}

Incident:
{evidence}

Deterministic runbooks:
{runbooks}

Hard rules:

1. First classify the incident as one of:
   - cosmetic non-root shell warning
   - service-readiness problem
   - platform/orchestrator lifecycle issue
   - deliberate operator/user termination

2. If the logs contain `user requested kill` and `exit code 137`,
   do NOT default to OOM. Treat it as likely SIGKILL from an explicit kill request unless contradicted by evidence.

3. If the service became available and accepted a public key after sshd warnings,
   treat `/run/sshd.pid` and login-record warnings as likely secondary unless they clearly blocked the session.

4. Do not invent Slurm, NCCL, or training-runtime failures unless the log actually shows them.

5. Every conclusion MUST reference a log line.

6. Commands must be real Linux/container/platform diagnostics.

Output EXACTLY:

## Incident Class
## Evidence Mapping (log → conclusion)
## Immediate Safe Actions
## Deep Diagnostics (exact commands)
## Exit Code / Termination Interpretation
## Recovery Path
## Prevention / Monitoring

Be precise.
No generic cloud advice.
No imaginary tools.
"""
    else:
        # More generic prompt (works for ANY HPC runtime issue, not only Slurm/DDP)
        prompt = f"""
You are a Tier-3 HPC Infrastructure SRE specializing in:

- NVIDIA GPUs
- Mellanox / mlx5 InfiniBand
- NCCL / PyTorch DDP
- multi-node training failures

You do NOT act like generic DevOps.
You act like NVIDIA Enterprise Support.

Environment:
{env_context}

Incident:
{evidence}

Deterministic runbooks:
{runbooks}

Hard rules:

1. Always reason in this order:
   Physical link → PCI device → kernel driver → subnet manager → NCCL.

2. NEVER assume systemd rdma service exists.

3. If InfiniBand appears:
   ALWAYS include:

   - ibstat
   - ibv_devinfo
   - lspci | grep -i mellanox
   - dmesg | grep mlx
   - sminfo (if available)

4. If GPU appears:
   ALWAYS include:

   - nvidia-smi
   - nvidia-smi topo -m
   - dmesg | grep Xid

5. Explain NCCL dependency on fabric.

6. Every conclusion MUST reference a log line.

7. Rank causes with confidence %.

8. Commands must be real Linux HPC commands.

9. For every command:
   explain expected output and interpretation.

10. Treat this as production outage.

Output EXACTLY:

## Root Cause (ranked with confidence %)
## Evidence Mapping (log → conclusion)
## Fabric / GPU State Analysis
## Immediate Safe Actions
## Deep Diagnostics (exact commands)
## Expected Results (how to interpret)
## Recovery Path
## Prevention / Monitoring

Be precise.
No generic cloud advice.
No imaginary tools.
No summaries.
"""

    client = InferenceClient(model=model, token=token)

    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
        )
        return res.choices[0].message.content
    except BadRequestError as e:
        # Common: model not chat-capable or not enabled on your HF providers
        return (
            "ERROR: HF Inference rejected this model request.\n\n"
            f"Details: {e}\n\n"
            "Fix:\n"
            "- pick another model from the dropdown, OR\n"
            "- enable a provider that serves that model, OR\n"
            "- use a chat-capable model on your enabled providers.\n"
        )
    except Exception as e:
        return f"ERROR: LLM request failed: {e}"


# ============================
# UI boundary wrappers
# ============================

def _ui_boundary_state(text: str, env_context: str = "") -> dict:
    incident = extract_incident(text)
    gaps = assess_evidence_gaps(text, env_context)
    scope = gaps["scope_assessment"]
    structure = gaps.get("log_structure") or analyze_log_structure(text)
    profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")
    high_priority_gaps = [gap["gap"] for gap in gaps["evidence_gaps"] if gap["priority"] == "high"]

    should_compact = profile in {"access_ops", "generic_ops"} or structure["mixed_sources"] or bool(high_priority_gaps)

    if structure["mixed_sources"]:
        reason = "Mixed transcript and embedded artifacts were detected."
        next_step = "Split the failing job log, scheduler output, and any comparison logs into separate artifacts, then rerun the advanced views."
    elif profile == "access_ops":
        reason = scope["explanation"]
        next_step = "Fix the access/bootstrap path first, or provide actual runtime logs from inside the cluster."
    elif profile == "generic_ops":
        reason = "The incident class is still too ambiguous for agentic routing."
        next_step = "Add adjacent logs, launch artifacts, and basic process/system context before using advanced views."
    elif high_priority_gaps:
        reason = high_priority_gaps[0]
        next_step = "Close the high-priority evidence gaps first, then retry the advanced views."
    else:
        reason = ""
        next_step = ""

    return {
        "should_compact": should_compact,
        "incident_id": incident["id"],
        "workflow_profile": profile,
        "reason": reason,
        "next_step": next_step,
        "top_signals": incident["top_signals"][:5],
        "high_priority_gaps": high_priority_gaps[:5],
        "log_structure": structure,
        "expected_artifacts": gaps["expected_artifacts"][:4],
    }


def _ui_compact_payload(feature: str, text: str, env_context: str = "", extra: dict | None = None) -> dict | None:
    boundary = _ui_boundary_state(text, env_context)
    if not boundary["should_compact"]:
        return None

    payload = {
        "status": "suppressed_in_ui",
        "feature": feature,
        "incident_id": boundary["incident_id"],
        "workflow_profile": boundary["workflow_profile"],
        "reason": boundary["reason"],
        "top_signals": boundary["top_signals"],
        "high_priority_gaps": boundary["high_priority_gaps"],
        "next_step": boundary["next_step"],
        "ui_note": "Raw agentic payloads are hidden in the UI until the evidence is grounded enough to trust them.",
    }
    if boundary["log_structure"]["mixed_sources"]:
        payload["log_structure"] = boundary["log_structure"]
    if extra:
        payload.update(extra)
    return payload


def _ui_join_values(items: list, limit: int = 5) -> str:
    values = []
    for item in items[:limit]:
        if isinstance(item, dict):
            values.append(str(item.get("value", item)))
        else:
            values.append(str(item))
    return ", ".join(values) if values else "none"


def _ui_boundary_markdown(
    feature: str,
    text: str,
    env_context: str = "",
    focus_lines: list[str] | None = None,
    top_signals_limit: int = 0,
) -> str | None:
    boundary = _ui_boundary_state(text, env_context)
    if not boundary["should_compact"]:
        return None

    lines = [
        f"**{feature} Deferred**",
        "",
        f"Reason: {boundary['reason']}",
    ]
    if focus_lines:
        lines.extend(["", *focus_lines])
    if boundary["high_priority_gaps"] and not focus_lines:
        lines.extend(
            [
                "",
                "Blocking gap:",
                *[f"- {gap}" for gap in boundary["high_priority_gaps"]],
            ]
        )
    if top_signals_limit:
        lines.extend(
            [
                "",
                "Relevant signals:",
                *[f"- {signal}" for signal in boundary["top_signals"][:top_signals_limit]],
            ]
        )
    lines.extend(["", f"Next step: {boundary['next_step']}"])
    return "\n".join(lines)


def build_incident_bundle_for_ui(text: str, env_context: str = "") -> str:
    boundary = _ui_boundary_state(text, env_context)
    if boundary["should_compact"]:
        return _ui_boundary_markdown(
            "Advanced Incident Bundle",
            text,
            env_context,
            focus_lines=[
                "The app will not build a full cross-artifact bundle from a pasted mixed transcript.",
                "Split the failing artifact first, then rerun bundle generation.",
            ],
            top_signals_limit=3,
        ) or ""

    bundle = build_incident_bundle(text, env_context)
    incident = bundle["incident"]
    scope = bundle["scope_assessment"]
    topology = bundle["topology"]
    structure = bundle["log_structure"]
    guidance = bundle["official_guidance"][:4]

    lines = [
        "**Advanced Incident Bundle**",
        "",
        f"Incident: `{incident['id']}`",
        f"Profile: `{scope.get('workflow_profile', 'runtime_ops')}`",
        f"Tags: {', '.join(incident['tags']) or 'none'}",
        f"Hosts: {_ui_join_values(topology['hosts'])}",
        f"Jobs: {_ui_join_values(topology['job_ids'])}",
        f"Ranks: {_ui_join_values(topology['ranks'], limit=8)}",
        f"GPUs: {_ui_join_values(topology['gpus'])}",
        f"Interfaces: {_ui_join_values(topology['network_interfaces'])}",
        f"Mixed sources: {'yes' if structure['mixed_sources'] else 'no'}",
        "",
        "Top signals:",
        *[f"- {signal}" for signal in incident["top_signals"][:6]],
    ]
    if guidance:
        lines.extend(
            [
                "",
                "Official guidance to consult:",
                *[f"- {item['title']}" for item in guidance],
            ]
        )
    return "\n".join(lines)


def parse_hpc_failure_artifacts_for_ui(text: str, env_context: str = "") -> str:
    signatures = parse_hpc_failure_artifacts(text)
    boundary = _ui_boundary_state(text, env_context)
    if boundary["should_compact"]:
        lines = [
            "**Structured Incident Signatures Deferred**",
            "",
            f"Reason: {boundary['reason']}",
            "",
            "Detected so far:",
            f"- Slurm: {'yes' if signatures['slurm']['detected'] else 'no'}"
            + (f" (jobs: {', '.join(signatures['slurm']['job_ids'][:3])})" if signatures["slurm"]["job_ids"] else ""),
            f"- NCCL: {'yes' if signatures['nccl']['detected'] else 'no'}"
            + (f" ({len(signatures['nccl']['transport_lines'])} transport lines)" if signatures["nccl"]["transport_lines"] else ""),
            f"- TorchElastic: {'yes' if signatures['torchelastic']['detected'] else 'no'}",
            f"- Platform/orchestrator: {'yes' if signatures['platform']['detected'] else 'no'}",
            "",
            "Next step: run signature parsing on the single failing artifact after you split the transcript.",
        ]
        return "\n".join(lines)

    lines = [
        "**Structured Incident Signatures**",
        "",
        f"- Slurm: {'detected' if signatures['slurm']['detected'] else 'not detected'}"
        + (f"; job ids `{', '.join(signatures['slurm']['job_ids'][:3])}`" if signatures["slurm"]["job_ids"] else ""),
        f"- NCCL: {'detected' if signatures['nccl']['detected'] else 'not detected'}"
        + (f"; sample lines `{len(signatures['nccl']['transport_lines'])}`" if signatures["nccl"]["transport_lines"] else ""),
        f"- TorchElastic: {'detected' if signatures['torchelastic']['detected'] else 'not detected'}",
        f"- Platform/orchestrator: {'detected' if signatures['platform']['detected'] else 'not detected'}",
    ]
    if signatures["nccl"]["transport_lines"]:
        lines.extend(
            [
                "",
                "Sample NCCL lines:",
                *[f"- {line}" for line in signatures["nccl"]["transport_lines"][:3]],
            ]
        )
    return "\n".join(lines)


def build_collection_manifest_for_ui(text: str, env_context: str = "") -> str:
    boundary = _ui_boundary_state(text, env_context)
    if boundary["should_compact"]:
        return "\n".join(
            [
                "**Collection Manifest Deferred**",
                "",
                f"Reason: {boundary['reason']}",
                "",
                "Collect first:",
                *[f"- {item['artifact']}" for item in boundary["expected_artifacts"][:4]],
                "",
                f"Next step: {boundary['next_step']}",
            ]
        )

    manifest = build_collection_manifest(text, env_context)
    lines = [
        "**Collection Manifest**",
        "",
        f"Ready for LLM: {'yes' if manifest['ready_for_llm'] else 'no'}",
        f"Grounding score: `{manifest['grounding_score']}`",
        f"Confidence: `{manifest['confidence_label']}`",
        "",
        "Expected artifacts:",
        *[f"- {item['artifact']}" for item in manifest["expected_artifacts"][:5]],
        "",
        "Filesystem next:",
        *[f"- {item['artifact']}" for item in manifest["filesystem"][:3]],
        "",
        "Shell next:",
        *[f"- `{item['command']}`" for item in manifest["shell"][:3]],
    ]
    return "\n".join(lines)


def suggest_mcp_companions_for_ui(text: str, env_context: str = "") -> str:
    boundary = _ui_boundary_state(text, env_context)
    if boundary["should_compact"]:
        scope = assess_incident_scope(text, env_context)
        profile = scope.get("workflow_profile", "runtime_ops" if scope["should_enable_hpc_workflows"] else "generic_ops")
        recommended_now = ["filesystem", "shell"]
        if profile == "runtime_ops" and any(tag in extract_incident(text)["tags"] for tag in ["NCCL", "GPU", "Scheduler"]):
            recommended_now.append("browser/docs")
        return "\n".join(
            [
                "**Companion MCP Routing Deferred**",
                "",
                f"Reason: {boundary['reason']}",
                "",
                "Use right now:",
                *[f"- `{server}`" for server in recommended_now],
                "",
                "Do not fan out into Playwright, telemetry, or a full multi-agent plan until the artifacts are split cleanly.",
            ]
        )

    companions = suggest_mcp_companions(text)
    lines = [
        "**Recommended Companion MCP Servers**",
        "",
        *[f"- `{item['server']}`: {item['why']}" for item in companions["recommended_servers"]],
    ]
    return "\n".join(lines)


def build_agentic_response_plan_for_ui(text: str, env_context: str = "") -> str:
    boundary = _ui_boundary_state(text, env_context)
    if boundary["should_compact"]:
        return "\n".join(
            [
                "**Agentic Response Plan Deferred**",
                "",
                "Current phase: `Grounding checkpoint`",
                "",
                "Do next:",
                "- Keep the failing job log separate from shell transcripts or comparison logs.",
                "- Collect the missing high-priority artifacts before asking for an agentic plan.",
                "- Retry the agentic plan once the upload is grounded.",
            ]
        )

    plan = build_agentic_response_plan(text, env_context)
    lines = [
        "**Agentic Response Plan**",
        "",
        f"Incident: `{plan['incident_id']}`",
        "",
        "Phases:",
        *[f"- {phase['phase']}: {phase['stop_when']}" for phase in plan["phases"]],
    ]
    return "\n".join(lines)


def generate_mcp_investigation_prompt_for_ui(text: str, env_context: str, objective: str) -> str:
    compact = _ui_boundary_markdown(
        "MCP Investigation Prompt",
        text,
        env_context,
        focus_lines=["No prompt is generated until the artifacts are split cleanly."],
    )
    return compact or generate_mcp_investigation_prompt(text, env_context, objective)


def generate_playwright_mission_for_ui(text: str, env_context: str = "", surface: str = "Grafana dashboard") -> str:
    compact = _ui_boundary_markdown(
        "Playwright Mission",
        text,
        env_context,
        focus_lines=["Browser work is intentionally skipped until the investigation has a clean artifact boundary."],
    )
    return compact or generate_playwright_mission(text, env_context, surface)


def suggest_fix_for_ui(logs: str, env_context: str, model: str) -> str:
    response = suggest_fix(logs, env_context, model)
    if response.startswith("ERROR: HF Inference rejected this model request."):
        return "\n".join(
            [
                "## Remote Remediation Unavailable",
                "",
                response.replace("ERROR: ", "", 1),
            ]
        )
    if response.startswith("ERROR: LLM request failed:"):
        return "\n".join(
            [
                "## Remote Remediation Unavailable",
                "",
                response.replace("ERROR: ", "", 1),
            ]
        )
    return response


# ============================
# UI
# ============================

CHAT_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

MCP_WORKFLOW_OBJECTIVES = [
    "Root cause deep dive",
    "Collect adjacent evidence with companion MCP servers",
    "Generate an escalation handoff",
    "Plan the next deterministic diagnostics",
]

DEFAULT_LOG_EXAMPLE = """2026-02-05 10:14:33 ERROR NCCL WARN Timeout waiting for rank 2
2026-02-05 10:14:34 CUDA out of memory. Tried to allocate 4096 MiB
Traceback (most recent call last):
  File "/workspace/train.py", line 198, in <module>
    loss.backward()
RuntimeError: CUDA error: out of memory

Connection refused to 10.148.92.21
User email: anouar@test.com
HF token hf_abcde12345678901234567890

NFS stale file handle on /mnt/vast/datasets
"""

DEFAULT_ENV_CLUSTER42 = """Cluster: Forty-Two (A100-based), ~79 nodes
Scheduler: (no Slurm) / internal orchestration
Interconnect: InfiniBand (bond0)
Storage: VAST (NFS) mounted on /mnt/vast
Workload: multi-node PyTorch DDP (nccl)
Container: (if any) <your image here>
Notes: common pain points: dist.barrier(), NCCL timeouts, NFS stale handles
"""

with gr.Blocks(title="Log Alchemist — Agentic Incident Response for HPC") as demo:
    gr.Markdown(
        "# 🧙 Log Alchemist — Agentic Incident Response for HPC\n"
        "Upload one or many log files, or paste logs directly. Uploading a screenshot will OCR automatically and fill the Logs box.\n\n"
        "**MCP tools exposed:** `extract_incident`, `assess_incident_scope`, `build_incident_bundle`, `parse_hpc_failure_artifacts`, `build_collection_manifest`, `open_incident_session`, `append_incident_artifact`, `assess_evidence_gaps`, `build_agentic_response_plan`, `suggest_mcp_companions`, `generate_playwright_mission`, `resolve_incident_session`, `find_similar_incidents`, `suggest_fix`.\n\n"
        "What makes this different from a generic chat upload: it preserves privacy, mines recurring log templates, distinguishes access issues from platform/orchestrator incidents and true runtime failures, parses artifacts from systems like TorchElastic, Slurm, and containerized shell platforms, builds local incident memory over time, and can orchestrate a stateful MCP investigation across filesystem, shell, Playwright, docs, and telemetry companions.\n\n"
        "Advanced agentic views stay compact in the UI until the upload is grounded enough to trust them."
    )

    with gr.Tabs():
        with gr.Tab("Log Triage"):
            with gr.Row():
                file = gr.File(
                    label="Upload one or many .log / .txt files (optional)",
                    file_types=[".log", ".txt"],
                    file_count="multiple",
                )
                img = gr.Image(type="pil", label="Upload screenshot of logs (optional)")

            logs = gr.Textbox(lines=16, label="Logs", value=DEFAULT_LOG_EXAMPLE)

            # Auto actions (no buttons)
            file.change(load_file_to_text, inputs=file, outputs=logs)
            img.change(ocr_image_to_text, inputs=img, outputs=logs)

            env = gr.Textbox(
                lines=8,
                label="Environment Context (VERY important for good suggestions)",
                value=DEFAULT_ENV_CLUSTER42,
            )

            with gr.Row():
                btn_json = gr.Button("Extract incident (JSON)")
                btn_scope = gr.Button("Assess scope")
                btn_card = gr.Button("Generate incident card (Markdown)")
                btn_bundle = gr.Button("Build advanced incident bundle")
                btn_signatures = gr.Button("Parse incident artifacts")
                btn_manifest = gr.Button("Build collection manifest")
                btn_save_memory = gr.Button("Save to local memory")
                btn_find_similar = gr.Button("Find similar incidents")
                btn_gaps = gr.Button("Assess evidence gaps")

            out_json = gr.JSON(label="Incident JSON")
            out_scope = gr.JSON(label="Incident Scope Assessment")
            out_md = gr.Markdown(label="Incident Card")
            out_bundle = gr.Markdown(label="Advanced Incident Bundle")
            out_signatures = gr.Markdown(label="Structured Incident Signatures")
            out_manifest = gr.Markdown(label="Collection Manifest")
            out_memory = gr.JSON(label="Local Incident Memory")
            out_gaps = gr.JSON(label="Evidence Gaps")

            btn_json.click(extract_incident, inputs=logs, outputs=out_json)
            btn_scope.click(assess_incident_scope, inputs=[logs, env], outputs=out_scope)
            btn_card.click(make_incident_card, inputs=logs, outputs=out_md)
            btn_bundle.click(build_incident_bundle_for_ui, inputs=[logs, env], outputs=out_bundle)
            btn_signatures.click(parse_hpc_failure_artifacts_for_ui, inputs=[logs, env], outputs=out_signatures)
            btn_manifest.click(build_collection_manifest_for_ui, inputs=[logs, env], outputs=out_manifest)
            btn_save_memory.click(save_incident_to_memory, inputs=[logs, env], outputs=out_memory)
            btn_find_similar.click(find_similar_incidents, inputs=logs, outputs=out_memory)
            btn_gaps.click(assess_evidence_gaps, inputs=[logs, env], outputs=out_gaps)

            gr.Markdown("## 🔬 Evidence-First Remediation\nDeterministic triage is shown immediately. Remote model synthesis is layered on only when the upload is grounded enough to trust.")
            model = gr.Dropdown(CHAT_MODELS, value=CHAT_MODELS[0], label="LLM (Chat Completion)")
            out_fix = gr.Markdown()

            gr.Button("Suggest root cause + fix steps (LLM)").click(
                suggest_fix_for_ui,
                inputs=[logs, env, model],
                outputs=out_fix,
            )

        with gr.Tab("Agentic MCP"):
            gr.Markdown(
                "This tab turns the project into an agentic incident-response conductor.\n"
                "Instead of a generic remote-client demo, it creates investigation plans, companion-tool routing, and Playwright missions that another MCP-capable agent can execute.\n\n"
                "If the upload is mixed or under-grounded, this tab returns a short blocker summary instead of a long mission."
            )

            workflow_objective = gr.Dropdown(
                MCP_WORKFLOW_OBJECTIVES,
                value=MCP_WORKFLOW_OBJECTIVES[0],
                label="Investigation objective",
            )
            playwright_surface = gr.Dropdown(
                PLAYWRIGHT_SURFACES,
                value="Grafana dashboard",
                label="Playwright surface",
            )
            workflow_companions = gr.Markdown(label="Recommended Companion MCP Servers")
            workflow_plan = gr.Markdown(label="Agentic Response Plan")
            workflow_prompt = gr.Markdown(label="Generated MCP Investigation Prompt")
            workflow_playwright = gr.Markdown(label="Generated Playwright Mission")

            with gr.Row():
                gr.Button("Suggest companion MCP servers").click(
                    suggest_mcp_companions_for_ui,
                    inputs=[logs, env],
                    outputs=workflow_companions,
                )
                gr.Button("Build agentic response plan").click(
                    build_agentic_response_plan_for_ui,
                    inputs=[logs, env],
                    outputs=workflow_plan,
                )
                gr.Button("Generate MCP investigation prompt").click(
                    generate_mcp_investigation_prompt_for_ui,
                    inputs=[logs, env, workflow_objective],
                    outputs=workflow_prompt,
                )
                gr.Button("Generate Playwright mission").click(
                    generate_playwright_mission_for_ui,
                    inputs=[logs, env, playwright_surface],
                    outputs=workflow_playwright,
                )

            gr.Markdown(
                "Recommended real-world pairing:\n"
                "- Log Alchemist MCP for HPC-specific parsing, bundle generation, and memory.\n"
                "- Filesystem MCP to open adjacent rank logs, configs, and saved diagnostics.\n"
                "- Shell/command MCP to run the next deterministic checks.\n"
                "- Playwright MCP to inspect platform UIs, Grafana, scheduler portals, Kibana, or cluster dashboards and capture browser evidence.\n"
                "- Browser/docs MCP to confirm Determined, NCCL, PyTorch Elastic, or Slurm guidance from official documentation."
            )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
