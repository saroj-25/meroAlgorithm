"""End-to-end checks over the built artefacts (skipped if not built yet)."""

import json

from src.data.build_knowledge_base import TABLE1


def test_knowledge_base_matches_table1(kb_path):
    chunks = [json.loads(line) for line in open(kb_path, encoding="utf-8")]
    assert len(chunks) == 1252
    assert sum(1 for c in chunks if c["language"] == "cm") == 184
    for topic, (n_en, n_cm) in TABLE1.items():
        topic_chunks = [c for c in chunks if c["topic"] == topic]
        assert len(topic_chunks) == n_en + n_cm, topic


def test_chunk_ids_are_unique(kb_path):
    chunks = [json.loads(line) for line in open(kb_path, encoding="utf-8")]
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_retrieval_returns_ranked_chunks(retriever):
    results = retriever.retrieve("what is the time complexity of merge sort", top_k=5)
    assert len(results) == 5
    assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]
    assert any("sort" in r["concept"] for r in results)


def test_code_mixed_query_retrieves_relevant_topic(retriever):
    results = retriever.retrieve("yo dijkstra negative weight ma kina fail huncha")
    assert any(r["topic"] == "Graphs and Traversals" for r in results)


def test_ablation_switches_change_results(retriever):
    dense = retriever.retrieve("kruskal minimum spanning tree", use_bm25=False, rerank=False)
    lexical = retriever.retrieve("kruskal minimum spanning tree", use_dense=False, rerank=False)
    assert dense and lexical
    assert all(r["bm25_score"] == 0 for r in dense)
    assert all(r["dense_score"] == 0 for r in lexical)


def test_generated_answer_is_grounded_and_cited(cfg, retriever):
    from src.inference.rag_pipeline import AlgoSathiRAG
    from src.models.generator import TemplateGenerator
    from src.preprocessing.language_id import RuleBasedLID

    rag = AlgoSathiRAG(cfg, retriever=retriever, generator=TemplateGenerator(),
                       detector=RuleBasedLID())
    response = rag.ask("what is the time complexity of merge sort")
    assert response.citations, "every answer must cite at least one chunk"
    retrieved = {c["chunk_id"] for c in response.chunks}
    assert all(c in retrieved for c in response.citations), "no fabricated citations"
    assert response.grounded


def test_register_is_preserved_for_nepali_queries(cfg, retriever):
    from src.inference.rag_pipeline import AlgoSathiRAG
    from src.models.generator import TemplateGenerator
    from src.preprocessing.language_id import RuleBasedLID

    rag = AlgoSathiRAG(cfg, retriever=retriever, generator=TemplateGenerator(),
                       detector=RuleBasedLID())
    response = rag.ask("binary search ko lower bound bhaneko k ho")
    assert response.query_type in {"code_mixed", "romanized_nepali"}
    # The Nepali scaffold must appear in the answer.
    assert any(marker in response.answer
               for marker in ["sajilo bhasama", "Kasari apply garne", "prashna tapailai"])
