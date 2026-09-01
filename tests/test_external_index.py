"""First-class canonical-host indexing and hard-scope retrieval."""

from __future__ import annotations

import importlib


def _modules():
    from memem import embedding_index, models, obsidian_store, operations, retrieve, search_index

    # Environment-backed paths are fixed at import time. Reload in dependency
    # order after the isolated vault/state fixtures set their environment.
    importlib.reload(models)
    importlib.reload(search_index)
    importlib.reload(embedding_index)
    importlib.reload(obsidian_store)
    importlib.reload(operations)
    importlib.reload(retrieve)
    return operations, obsidian_store, retrieve, search_index


def test_external_record_indexes_abstraction_and_cues_not_full_value(
    tmp_vault, tmp_cortex_dir, monkeypatch,
):
    operations, store, retrieve, search = _modules()
    monkeypatch.setattr(retrieve, "_get_model", lambda: None)
    monkeypatch.setattr("memem.embedding_index._get_model", lambda: None)

    first = operations.memory_index_upsert(
        "knowledge-parser-bounds",
        "The verified fix contains incidentalpayload that must stay pull-only.",
        primary_index="Parser bounds regression fix",
        cues=["parser boundary", "request validation"],
        scope_id="project-hunter",
        title="Parser bounds fix",
        tags="fact,validated",
        paths=["src/parser.py"],
    )
    second = operations.memory_index_upsert(
        "knowledge-parser-bounds",
        "Updated evidence still contains incidentalpayload and full detail.",
        primary_index="Parser bounds regression fix",
        cues=["parser boundary", "request validation"],
        scope_id="project-hunter",
        title="Parser bounds fix",
        tags="fact,validated",
        paths=["src/parser.py"],
    )

    assert first["id"] == second["id"]  # external identity, not fuzzy dedup
    assert second["external_id"] == "knowledge-parser-bounds"
    assert second["primary_index"] == "Parser bounds regression fix"
    assert "parser boundary" in store._embedding_value(second)
    assert "incidentalpayload" not in store._embedding_value(second)
    assert len(list((tmp_vault / "memem" / "memories").glob("*.md"))) == 1

    # The primary abstraction and cue anchors are searchable; incidental detail
    # in the high-fidelity body is intentionally not an FTS anchor.
    assert search._search_fts("parser bounds", scope_id="project-hunter") == [second["id"]]
    assert search._search_fts("parser boundary", scope_id="project-hunter") == [second["id"]]
    assert search._search_fts("incidentalpayload", scope_id="project-hunter") == []


def test_hard_scope_filters_before_ranking_and_remove_retires_projection(
    tmp_vault, tmp_cortex_dir, monkeypatch,
):
    operations, store, retrieve, _search = _modules()
    monkeypatch.setattr(retrieve, "_get_model", lambda: None)
    monkeypatch.setattr("memem.embedding_index._get_model", lambda: None)

    hunter = operations.memory_index_upsert(
        "hunter-parser",
        "Hunter detail",
        primary_index="Parser bounds regression",
        cues=["parser boundary"],
        scope_id="project-hunter",
    )
    operations.memory_index_upsert(
        "other-parser",
        "Other project detail with many parser parser parser tokens",
        primary_index="Parser bounds regression",
        cues=["parser boundary"],
        scope_id="project-other",
    )

    hits = retrieve.retrieve(
        "parser bounds", k=8, scope_id="project-hunter", scope_mode="hard",
        log_call_type=None, writeback=False,
    )
    assert [hit["external_id"] for hit in hits] == ["hunter-parser"]
    assert all(hit["project"] == "project-hunter" for hit in hits)

    assert operations.memory_index_remove("hunter-parser") is True
    assert retrieve.retrieve(
        "parser bounds", k=8, scope_id="project-hunter", scope_mode="hard",
        log_call_type=None, writeback=False,
    ) == []
    retired = store._find_memory(hunter["id"])
    assert retired is not None and retired["status"] == "deprecated"


def test_external_host_can_index_a_valid_short_canonical_value(
    tmp_vault, tmp_cortex_dir, monkeypatch,
):
    operations, _store, retrieve, _search = _modules()
    monkeypatch.setattr(retrieve, "_get_model", lambda: None)
    monkeypatch.setattr("memem.embedding_index._get_model", lambda: None)

    result = operations.memory_index_upsert(
        "preference-tabs", "Use tabs", primary_index="Editor indentation preference",
        cues=["tabs indentation"], scope_id="user-local",
    )
    assert result["essence"] == "Use tabs"
    assert result["external_id"] == "preference-tabs"
