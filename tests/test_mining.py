"""Tests for mining pipeline (mocked Haiku)."""



def test_extract_json_string():
    from memem.mining import _extract_json_string
    assert _extract_json_string('[{"title": "test"}]') == '[{"title": "test"}]'
    assert _extract_json_string('prefix [{"title": "x"}] suffix') == '[{"title": "x"}]'
    assert _extract_json_string('no json here') is None


def test_repair_json_bracket_in_string():
    """_repair_json must skip over string literals so a title like
    ``"see [note"`` doesn't fool the bracket counter into appending a
    spurious closer that produces invalid JSON."""
    import json

    from memem.mining import _repair_json

    balanced = '[{"title": "see [note", "content": "ok"}]'
    assert _repair_json(balanced) == balanced
    assert json.loads(_repair_json(balanced)) == [{"title": "see [note", "content": "ok"}]

    truncated_outer = '[{"title": "see [note", "content": "ok"}'
    assert json.loads(_repair_json(truncated_outer)) == [{"title": "see [note", "content": "ok"}]

    # Mid-string truncation: also closes the string
    truncated_mid = '[{"title": "truncated pr'
    parsed = json.loads(_repair_json(truncated_mid))
    assert parsed == [{"title": "truncated pr"}]


def test_normalize_scope_id_alias():
    """Aliases normalize to canonical names so consolidation doesn't skip
    a project tagged under a pre-rename alias (e.g. 'cortex' → 'cortex-plugin')."""
    from memem.models import _normalize_scope_id
    assert _normalize_scope_id("cortex") == "cortex-plugin"
    assert _normalize_scope_id("default") == "general"
    assert _normalize_scope_id("") == "general"
    assert _normalize_scope_id("substrate") == "substrate"


# ---------------------------------------------------------------------------
# C2: keys validation in _mine_one_chunk
# ---------------------------------------------------------------------------

def test_mine_one_chunk_keys_cap_and_sanitize(monkeypatch):
    """_mine_one_chunk caps keys to 8 items and 60 chars each, discards non-strings."""
    import json
    from memem import mining

    raw_keys = [
        "short",
        "a" * 70,   # over 60 chars — should be truncated to 60
        123,         # non-string — should be discarded
        None,        # non-string — should be discarded
        "auth",
        "oauth2",
        "pkce",
        "openid",
        "jwt",       # 9th item — should be dropped (cap 8)
    ]
    haiku_output = json.dumps([{
        "title": "OAuth keys test",
        "content": "Details about OAuth flow with PKCE",
        "project": "general",
        "importance": 3,
        "keys": raw_keys,
    }])

    monkeypatch.setattr(mining, "_run_haiku", lambda _prompt: haiku_output)
    results = mining._mine_one_chunk(["some conversation text"])

    assert len(results) == 1
    entry = results[0]
    keys = entry["keys"]
    assert isinstance(keys, list), "keys should be a list"
    # Non-strings (123, None) discarded; remaining: short, a*70, auth, oauth2, pkce, openid, jwt (cap 8 means first 8 str)
    # After discarding non-strings: short(1), a*70(2), auth(3), oauth2(4), pkce(5), openid(6), jwt(7) = 7 items → all fit in 8
    assert len(keys) <= 8, f"Expected ≤8 keys, got {len(keys)}"
    # Check that the long key was truncated to 60
    for k in keys:
        assert len(k) <= 60, f"Key '{k[:20]}...' exceeds 60 chars"
        assert isinstance(k, str), f"Key should be str, got {type(k)}"
    # Numeric/None items discarded
    assert 123 not in keys
    assert None not in keys
    assert "short" in keys


def test_mine_one_chunk_keys_missing_gives_empty(monkeypatch):
    """_mine_one_chunk sets keys=[] when the field is missing from Haiku output."""
    import json
    from memem import mining

    haiku_output = json.dumps([{
        "title": "No keys field",
        "content": "Some content without keys",
        "project": "general",
        "importance": 3,
        # No "keys" field
    }])

    monkeypatch.setattr(mining, "_run_haiku", lambda _prompt: haiku_output)
    results = mining._mine_one_chunk(["conversation"])
    assert results[0]["keys"] == []


def test_mine_one_chunk_keys_null_gives_empty(monkeypatch):
    """_mine_one_chunk sets keys=[] when keys is explicitly null."""
    import json
    from memem import mining

    haiku_output = json.dumps([{
        "title": "Null keys field",
        "content": "Some content with null keys",
        "project": "general",
        "importance": 3,
        "keys": None,
    }])

    monkeypatch.setattr(mining, "_run_haiku", lambda _prompt: haiku_output)
    results = mining._mine_one_chunk(["conversation"])
    assert results[0]["keys"] == []


def test_mine_one_chunk_keys_over_8_capped(monkeypatch):
    """_mine_one_chunk caps keys to max 8 items."""
    import json
    from memem import mining

    haiku_output = json.dumps([{
        "title": "Many keys",
        "content": "Content with too many keys",
        "project": "general",
        "importance": 3,
        "keys": [f"key{i}" for i in range(15)],  # 15 keys, should be capped to 8
    }])

    monkeypatch.setattr(mining, "_run_haiku", lambda _prompt: haiku_output)
    results = mining._mine_one_chunk(["conversation"])
    assert len(results[0]["keys"]) == 8


def test_mine_one_chunk_passes_episodic_kind(monkeypatch):
    """kind='episodic' (dated events) must survive validation alongside 'procedural'.

    Regression for the episodic-generation overhaul: the concept miner emits
    kind='episodic' for significant dated events, which mine_delta maps to a
    type:episodic tag so 'when did X happen' queries can find them.
    """
    import json

    from memem import mining

    haiku_output = json.dumps([
        {"title": "Switched to DeepSeek API", "content": "On June 8 switched backend to DeepSeek for streaming + cost",
         "project": "substrate", "importance": 4, "kind": "episodic"},
        {"title": "Always await sample loads", "content": "When using Strudel, always await sample loads",
         "project": "loom", "importance": 3, "kind": "procedural"},
        {"title": "Some concept", "content": "A durable convention", "project": "general",
         "importance": 3, "kind": "bogus"},
    ])
    monkeypatch.setattr(mining, "_run_haiku", lambda _prompt: haiku_output)
    results = mining._mine_one_chunk(["conversation text"])
    assert results[0].get("kind") == "episodic"
    assert results[1].get("kind") == "procedural"
    assert "kind" not in results[2]  # unrecognized kinds dropped


def test_merge_memories_rejects_empty_input():
    """Empty side must raise (no subprocess) — never store a 'please provide entries' reply."""
    import pytest

    from memem import mining
    with pytest.raises(RuntimeError, match="non-empty"):
        mining._merge_memories("", "some new content")
    with pytest.raises(RuntimeError, match="non-empty"):
        mining._merge_memories("existing content", "   ")


def test_merge_memories_rejects_haiku_refusal(monkeypatch):
    """A Haiku clarification reply (returncode 0) must raise, not be stored as content.

    Regression for the empty strudel merge-stub eeadd6c0, whose body was
    'I don't see two specific memory entries...'.
    """
    import subprocess

    import pytest

    from memem import mining

    refusal = ("I don't see two specific memory entries provided in your message. "
               "Could you please share the two memory entries you'd like me to merge?")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=refusal, stderr=""),
    )
    with pytest.raises(RuntimeError, match="clarification"):
        mining._merge_memories("real existing essence", "real new content")
