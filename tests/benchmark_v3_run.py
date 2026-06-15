#!/usr/bin/env python3
"""memem vs EverMe — v3 benchmark RUNNER (episodic-fair capture phase).

Fixes v2's bias: v2's "temporal" queries targeted events so recent neither store
had mined them (both scored ~0 = corpus-recency, not episodic skill). v3 adds a
proper EPISODIC category of dated "what happened / when / in what order" queries
on events that are WELL-MINED in BOTH stores (early-to-mid June work), so the
temporal-recall dimension — EverMe's claimed strength — is tested fairly.

Reuses v2's dual-system capture helpers; only the query set changes.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MEMEM_TELEMETRY_SOURCE", "benchmark")

import importlib.util as _u
_spec = _u.spec_from_file_location("bv2", str(Path(__file__).parent / "benchmark_v2_run.py"))
bv2 = _u.module_from_spec(_spec); _spec.loader.exec_module(bv2)

# type, query, gold_criterion, answerable
QUERIES = [
    # --- EPISODIC (dated event recall + temporal reasoning; all well-mined in both stores) ---
    ("episodic", "When did the user switch from the Claude CLI to the DeepSeek API, and what drove it?", "around June 8; switched for true char/token-level SSE streaming + ~20x lower cost", True),
    ("episodic", "What did the user work on in the session where the strudel/loom audio wasn't playing?", "diagnosing/fixing strudel sample loading + audio context (hh sound, doughSamples, await samples)", True),
    ("episodic", "List the memem releases through early-to-mid June, roughly in order.", "the v2.x sequence: ~v2.1.0, v2.2.0, v2.3.0, v2.4.0, v2.5.0, v2.6.0, v2.7.0, v2.8.0, v2.9.x", True),
    ("episodic", "What happened in the session that began with tmux crashing twice?", "user asked to recover/fetch latest sessions after tmux crash (session 9612f54c context)", True),
    ("episodic", "Around when, and how, did the user set up SSH access to the VPS?", "early-mid June; SSH tunnel + 'lexie' alias, ServerAliveInterval keepalive, -L port-forward", True),
    ("episodic", "What was the user exploring when they considered copying EverMe's architecture into memem?", "whether wholesale copy of EverMe was viable; concluded infeasible due to architectural mismatch (server-backed vs local)", True),
    # --- single-fact (answerable from both) ---
    ("single-fact", "What was memem renamed from, and in which version?", "renamed from cortex (cortex-plugin) in v0.7.0", True),
    ("single-fact", "What is memem's default injection mode since v2.4.0?", "MEMEM_INJECTION_MODE=tool (not auto)", True),
    ("single-fact", "What is the HFT arb strategy's risk architecture?", "pre-computed risk object (buyPrice+size) passed to executeArbEntry", True),
    # --- synthesis ---
    ("synthesis", "How does memem's mining pipeline work end to end?", "Stop hook -> detached mine_delta -> Haiku extract -> reconcile (ADD/UPDATE/SUPERSEDE/NOOP/PROFILE) -> vault", True),
    ("synthesis", "What is the forge worktree-clobber problem and its fix?", "parallel-tier worktrees branch from stale HEAD; fix = WIP-commit between tiers", True),
    ("synthesis", "Summarize the memem vs EverMe architectural trade-off.", "memem concept-vault local cheap/low-latency; EverMe event-log remote, strong episodic, higher token/latency", True),
    # --- preference / convention ---
    ("preference", "What is the user's git push convention for memem?", "push to private origin by default; public only when explicitly asked", True),
    ("preference", "How does the user want progress reported during long tasks?", "running progress updates (~25/50/done), not batched at the end", True),
    ("preference", "What is the user's stance on silent error handling?", "silent error handling non-negotiable; explicit logging required", True),
    # --- cross-project ---
    ("cross-project", "What is lexie and how is it different from memem?", "lexie = personalized recommendation engine over chat history; separate from memem", True),
    ("cross-project", "Which coding agents must vibereader support?", "Claude Code and Codex, with extensible plugin support for others", True),
    ("cross-project", "How does vibereader store its configuration?", "config file with storage locations / nested keys; project-level config", True),
    # --- abstention / negative (no valid answer) ---
    ("abstention", "What is the user's favorite pizza topping?", "NO ANSWER — never discussed; correct behavior is to surface nothing relevant", False),
    ("abstention", "What did the user decide about migrating memem to a Postgres backend?", "NO ANSWER — never happened; correct behavior is to surface nothing relevant", False),
]


def main():
    from memem.retrieve import retrieve
    retrieve("warmup query")
    out = []
    for typ, q, gold, answerable in QUERIES:
        rec = {"type": typ, "query": q, "gold": gold, "answerable": answerable}
        try:
            rec["memem_items"], rec["memem_lat_ms"], rec["memem_tok"] = bv2.run_memem(q)
        except Exception as e:  # noqa: BLE001
            rec["memem_items"], rec["memem_lat_ms"], rec["memem_tok"], rec["memem_err"] = [], 0, 0, str(e)[:80]
        try:
            rec["everme_items"], rec["everme_lat_ms"], rec["everme_tok"] = bv2.run_everme(q)
        except Exception as e:  # noqa: BLE001
            rec["everme_items"], rec["everme_lat_ms"], rec["everme_tok"], rec["everme_err"] = [], 0, 0, str(e)[:80]
        out.append(rec)
        print(f"  [{typ:<13}] memem {len(rec['memem_items'])}/{rec['memem_lat_ms']:.0f}ms  everme {len(rec['everme_items'])}/{rec['everme_lat_ms']:.0f}ms  {q[:40]}")
    json.dump(out, open("/tmp/bench_v3_results.json", "w"), indent=2)
    print(f"\n{len(out)} queries -> /tmp/bench_v3_results.json")


if __name__ == "__main__":
    main()
