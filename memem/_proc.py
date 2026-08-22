"""_proc — resolve the Claude CLI into an argv that actually starts on this OS.

On Linux and macOS ``subprocess.run(["claude", ...])`` works because npm
installs a real executable shim. On Windows npm installs ``claude.cmd``, and
``CreateProcess`` cannot start batch files — the call fails with::

    [WinError 2] The system cannot find the file specified

memem logs that as a warning and carries on, so ``--mine-session`` exits 0,
prints ``{"mined": "<id>"}`` and stores nothing. The session is then marked as
mined, so the transcript is never revisited. Silent data loss, and nothing in
the default output hints at it.

Going through ``cmd.exe /c`` would start the batch file, but cmd parses ``%``,
``&`` and ``^`` in its arguments — and we pass a multi-line system prompt as an
argument. Running the packaged Node entry point directly avoids the shell
entirely, so the prompt reaches the CLI byte for byte.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def claude_argv(args: list[str]) -> list[str]:
    """Return the argv to run ``claude`` with ``args`` on this platform."""
    exe = shutil.which("claude")
    if exe is None:
        # Let the caller fail with the usual "not found" so capability checks
        # and error messages stay unchanged.
        return ["claude", *args]

    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        cli = (
            Path(exe).parent
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "cli-wrapper.cjs"
        )
        if cli.is_file():
            return [shutil.which("node") or "node", str(cli), *args]

    return [exe, *args]
