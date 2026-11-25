"""Shared helper: run one tool-less `claude -p` call, return stdout lines."""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SANDBOX = HERE / "sandbox"


def claude(prompt, model="sonnet", system="You write training data. Output only JSONL lines, nothing else.",
           timeout=600):
    SANDBOX.mkdir(exist_ok=True)
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--system-prompt", system]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                       timeout=timeout, cwd=str(SANDBOX))
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode} {r.stderr.strip()[:200]} {r.stdout.strip()[:200]}")
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]
