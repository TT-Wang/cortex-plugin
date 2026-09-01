#!/usr/bin/env python3
"""Ground-truth recall@k benchmark — the honest companion to benchmark_18q.py.

benchmark_18q measures keyword-overlap precision@~8, which (a) has no ground truth,
(b) is inflated by edge-query keyword noise, and (c) contains unsatisfiable keywords
(e.g. 'f-001'/'f-002' never appear in any source transcript). This benchmark instead
labels the genuinely-correct memory id(s) per query and measures whether retrieval
surfaces one of them in the top-k — i.e. real recall, not substring luck.

GT ids were assigned by query INTENT and verified against the vault (titles inspected).
Edge/noise queries (no meaningful ground truth) are excluded. A query with an EMPTY GT
set is a known corpus-gap (honest miss until the memory is recovered).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMEM_TELEMETRY_SOURCE", "benchmark")
from memem.retrieve import retrieve

# query -> set of acceptable correct-memory id8s (any in top-k = recall hit). {} = corpus-gap.
GROUND_TRUTH = {
    ("episodic", "What did I work on yesterday related to memem v1.13.0?"): {"48c3b954"},
    ("episodic", "what was the v1.12.0 release scope"): {"4b06cf8d"},
    ("episodic", "what did we discuss about EverMe extraction mechanism"): {"27fa093f", "06df8993", "072681bb"},
    ("skill", "how to debug a stuck forge worker"): {"8d8b41da", "680aae9d"},
    ("skill", "how to mine session JSONL files"): {"2b08d109", "5e1e6396"},
    ("skill", "how do I push to private origin only"): {"962bb0e7", "750784f6", "68cd91f1"},
    ("case", "how did we fix the v1.12.0 active_memory_slice import bug"): {"54a19465", "c79d4a62", "77193dc9"},
    ("case", "what was the resolution for the strudel hh sound not loading"): set(),  # CORPUS GAP (stub only)
    ("case", "how was forge worktree clobber prevented across tiers"): {"0c1b514d", "d62b5fc8", "bf08025e", "a925f55b"},
    ("cross-scope", "what SSH command do I use for my VPS lexie"): {"c7a897ba", "a3ac5ce1", "29ddbac2"},
    ("cross-scope", "how does vibereader config validation work"): {"28720e11", "3b215793", "34001654"},
    ("cross-scope", "what is the architecture of HFT arb strategy"): {"f4ab7083", "e1a2eed4", "2e2c0933", "cd2a4d7b"},
    ("identity", "what is the cortex-plugin tech stack"): {"6d971ab0", "a35ad32f", "736515e6"},
    ("identity", "what does memem do"): {"c102b100", "6611a7ad"},
    ("identity", "where is the obsidian vault located"): {"eda7d9f5", "dcacc465", "e8eb48c1"},
}


def recall_at(k):
    retrieve("warmup query")
    hits = gaps = 0
    rows = []
    for (cat, q), gt in GROUND_TRUTH.items():
        if not gt:
            rows.append((cat, q, "GAP", "—")); gaps += 1; continue
        ids = [h.get("id", "")[:8] for h in (retrieve(q, k=k) or [])]
        found = next((i + 1 for i, x in enumerate(ids) if x in gt), None)
        if found:
            hits += 1; rows.append((cat, q, "hit", f"rank {found}"))
        else:
            rows.append((cat, q, "MISS", f">{k}"))
    return hits, gaps, rows


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    hits, gaps, rows = recall_at(K)
    n = len(GROUND_TRUTH)
    print(f"=== recall@{K} (ground-truth, {n} meaningful queries; edge/noise excluded) ===")
    for cat, q, status, rank in rows:
        mark = {"hit": "  ", "MISS": "✗ ", "GAP": "▢ "}[status]
        print(f"  {mark}[{cat:<11}] {status:<4} {rank:<8} {q[:46]}")
    print(f"\n  recall@{K} = {hits}/{n} = {100*hits/n:.1f}%   ({gaps} corpus-gap, counted as misses)")
    print(f"  recall@{K} on findable (excl. {gaps} gaps) = {hits}/{n-gaps} = {100*hits/max(1,n-gaps):.1f}%")
