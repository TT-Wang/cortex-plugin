#!/usr/bin/env python3
"""memem vs EverMe — v2 benchmark RUNNER (capture phase).

Fair, ground-truth, LLM-judge-ready benchmark. Fixes the old keyword benchmark's
flaws: no keyword-overlap scoring, no unsatisfiable keywords, includes abstention
queries, queries drawn from the SHARED corpus both systems mined.

This script only CAPTURES raw top-k results + latency + token cost from each
system into JSON; an LLM-judge workflow scores relevance afterwards.
"""
import json, os, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MEMEM_TELEMETRY_SOURCE", "benchmark")

K = 8

# type, query, gold_criterion (what a fully-correct answer must contain), answerable
QUERIES = [
    # --- single-fact recall ---
    ("single-fact", "What SSH alias and port-forward does the user use to reach the VPS?", "ssh alias 'lexie'; port-forward -L 5173:localhost:5173", True),
    ("single-fact", "What was memem renamed from, and in which version?", "renamed from cortex (cortex-plugin) in v0.7.0", True),
    ("single-fact", "Which AI API did the user switch to for streaming code generation and why?", "DeepSeek API; fine-grained char/token streaming via SSE delta.content; ~20x cheaper", True),
    ("single-fact", "What is memem's default injection mode since v2.4.0?", "MEMEM_INJECTION_MODE=tool (not auto)", True),
    # --- temporal / recency ---
    ("temporal", "What is the most recent memem release and its headline feature?", "v2.9.1 — path-scope activation (recent_session_paths feeds paths_context)", True),
    ("temporal", "What did the user work on for the strudel audio bug?", "strudel hh / sample-loading audio fix (await samples, doughSamples)", True),
    ("temporal", "What was recently decided about memem's graph subsystem?", "graph.db ablation: ~0.3% usage, delete/deprecate candidate; benchmark graph-blind", True),
    # --- multi-session synthesis ---
    ("synthesis", "How does memem's mining pipeline work end to end?", "Stop hook -> detached mine_delta subprocess -> Haiku extract -> reconcile (ADD/UPDATE/SUPERSEDE/NOOP/PROFILE) -> vault", True),
    ("synthesis", "Summarize the memem vs EverMe architectural trade-off.", "memem concept-vault local cheap/low-latency; EverMe event-log remote, strong episodic, higher token/latency", True),
    ("synthesis", "What is the forge worktree-clobber problem and its fix?", "parallel-tier worktrees branch from stale HEAD; fix = WIP-commit between tiers", True),
    # --- preference / convention ---
    ("preference", "What is the user's git push convention for memem?", "push to private origin by default; public only when explicitly asked", True),
    ("preference", "How does the user want progress reported during long tasks?", "running progress updates (~25/50/done), not batched at the end", True),
    ("preference", "What is the user's stance on silent error handling?", "silent error handling non-negotiable; explicit logging required", True),
    ("preference", "Does the user prefer the Claude CLI or direct API, and why?", "Claude CLI over API for cost (subscription vs metered)", True),
    # --- cross-project ---
    ("cross-project", "What is lexie and how is it different from memem?", "lexie = personalized recommendation engine over chat history; separate from memem", True),
    ("cross-project", "What is the HFT arb strategy's risk architecture?", "pre-computed risk object (buyPrice+size) passed to executeArbEntry", True),
    ("cross-project", "Which coding agents must vibereader support?", "Claude Code and Codex, with extensible plugin support for others", True),
    # --- abstention / negative (NO valid answer exists) ---
    ("abstention", "What is the user's preferred Kubernetes ingress controller?", "NO ANSWER — never discussed; correct behavior is to surface nothing relevant", False),
    ("abstention", "What did we decide about rewriting memem in Rust?", "NO ANSWER — never happened; correct behavior is to surface nothing relevant", False),
    ("abstention", "What is the user's favorite pizza topping?", "NO ANSWER — never discussed; correct behavior is to surface nothing relevant", False),
]


def run_memem(q):
    from memem.retrieve import retrieve
    t = time.monotonic()
    res = retrieve(q, k=K) or []
    lat = (time.monotonic() - t) * 1000
    items = [{"title": h.get("title", ""), "text": (h.get("body", "") or "")[:400]} for h in res]
    payload = sum(len(i["title"]) + len(i["text"]) for i in items)
    return items, round(lat, 0), payload


def _everme_cfg():
    cfg = {}
    p = Path.home() / ".claude" / "everme.env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); cfg[k.strip()] = v.strip()
    base = cfg.get("EVERME_API_BASE", "https://api.everme.evermind.ai").rstrip("/")
    return base, cfg.get("EVERME_AGENT_TOKEN") or cfg.get("EVERME_API_KEY", "")


def run_everme(q):
    base, token = _everme_cfg()
    url = base + ("/api/v1" if not base.endswith("/api/v1") else "") + "/mem/search"
    body = json.dumps({"query": q[:1024], "topK": K}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.loads(r.read())
    lat = (time.monotonic() - t) * 1000
    items = [{"title": it.get("subject", ""), "text": (it.get("episode", "") or "")[:400]}
             for it in (data.get("result") or {}).get("items", [])]
    payload = sum(len(i["title"]) + len(i["text"]) for i in items)
    return items, round(lat, 0), payload


def main():
    from memem.retrieve import retrieve
    retrieve("warmup query")
    out = []
    for typ, q, gold, answerable in QUERIES:
        rec = {"type": typ, "query": q, "gold": gold, "answerable": answerable}
        try:
            rec["memem_items"], rec["memem_lat_ms"], rec["memem_tok"] = run_memem(q)
        except Exception as e:  # noqa: BLE001
            rec["memem_items"], rec["memem_lat_ms"], rec["memem_tok"], rec["memem_err"] = [], 0, 0, str(e)[:80]
        try:
            rec["everme_items"], rec["everme_lat_ms"], rec["everme_tok"] = run_everme(q)
        except Exception as e:  # noqa: BLE001
            rec["everme_items"], rec["everme_lat_ms"], rec["everme_tok"], rec["everme_err"] = [], 0, 0, str(e)[:80]
        out.append(rec)
        print(f"  [{typ:<13}] memem {len(rec['memem_items'])} items {rec['memem_lat_ms']:.0f}ms / everme {len(rec['everme_items'])} items {rec['everme_lat_ms']:.0f}ms  {q[:42]}")
    json.dump(out, open("/tmp/bench_v2_results.json", "w"), indent=2)
    mt = sum(r.get("memem_tok", 0) for r in out) // 4 // len(out)
    et = sum(r.get("everme_tok", 0) for r in out) // 4 // len(out)
    ml = sum(r.get("memem_lat_ms", 0) for r in out) / len(out)
    el = sum(r.get("everme_lat_ms", 0) for r in out) / len(out)
    print(f"\n{len(out)} queries captured -> /tmp/bench_v2_results.json")
    print(f"  mean latency: memem {ml:.0f}ms (local) | everme {el:.0f}ms (network)")
    print(f"  mean payload: memem ~{mt} tok/q | everme ~{et} tok/q")


if __name__ == "__main__":
    main()
