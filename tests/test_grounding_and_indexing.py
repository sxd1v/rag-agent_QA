import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.agent.react_loop import execute_tool, should_finish
from app.agent.react_state import ReActState
from app.agent.tools import GenerateAnswerTool, REFUSAL_ANSWER
from app.core import cache
from app.core.config import KNOWLEDGE_COLLECTION_NAME, MEMORY_COLLECTION_NAME
from app.db import vector_store
from app.services import hybrid_retriever
from app.services.indexer import add_chunk_ids, index_documents


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLlm:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt):
        return FakeResponse(json.dumps(self.payload, ensure_ascii=False))


class AnswerGroundingTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document(
            page_content="RAG 会先检索证据再生成回答。",
            metadata={"chunk_id": "chunk-valid", "source": "guide.md"},
        )

    @patch("app.agent.tools.cache_set")
    @patch("app.agent.tools.cache_get", return_value=None)
    def test_answer_with_valid_context_citation_is_returned(self, _cache_get, _cache_set):
        llm = FakeLlm({
            "supported": True,
            "answer": "RAG 先检索再生成。[chunk-valid]",
            "citations": ["chunk-valid"],
            "missing": "",
        })
        with patch("app.agent.tools.get_chat_llm", return_value=llm):
            result = GenerateAnswerTool().execute("什么是 RAG？", [self.doc])

        self.assertFalse(result["abstained"])
        self.assertEqual(result["citations"], ["chunk-valid"])
        self.assertEqual(result["sources"][0]["source"], "guide.md")

    @patch("app.agent.tools.cache_set")
    @patch("app.agent.tools.cache_get", return_value=None)
    def test_answer_with_unverifiable_citation_is_refused(self, _cache_get, _cache_set):
        llm = FakeLlm({
            "supported": True,
            "answer": "一个没有证据的结论。[chunk-missing]",
            "citations": ["chunk-missing"],
            "missing": "",
        })
        with patch("app.agent.tools.get_chat_llm", return_value=llm):
            result = GenerateAnswerTool().execute("问题", [self.doc])

        self.assertTrue(result["abstained"])
        self.assertEqual(result["answer"], REFUSAL_ANSWER)
        self.assertEqual(result["citations"], [])

    @patch("app.agent.tools.cache_set")
    @patch("app.agent.tools.cache_get", return_value=None)
    def test_repeated_retrieval_forces_grounded_finalization(self, _cache_get, _cache_set):
        state = ReActState()
        state.reset("什么是 RAG？", retrieval_strategy="hybrid")
        llm = FakeLlm({
            "supported": True,
            "answer": "RAG 先检索再生成。[chunk-valid]",
            "citations": ["chunk-valid"],
            "missing": "",
        })
        with (
            patch("app.agent.tools.search_docs", return_value=[self.doc]),
            patch("app.agent.tools.get_chat_llm", return_value=llm),
        ):
            execute_tool("search_docs", {"query": state.query}, state)
            execute_tool("search_docs", {"query": state.query}, state)
            self.assertTrue(should_finish(state))

        self.assertEqual(state.no_progress_attempts, 1)
        self.assertEqual(state.final_citations, ["chunk-valid"])


class IndexConsistencyTests(unittest.TestCase):
    def test_retrieval_cache_clear_preserves_session_namespace(self):
        with patch.object(cache, "_redis_client", None):
            cache._memory_cache.clear()
            cache._memory_cache.update({
                "retrieval:hybrid:q:3": ["stale"],
                "session:user-1": [{"answer": "keep"}],
            })
            cache.clear_prefix("retrieval:")
            self.assertNotIn("retrieval:hybrid:q:3", cache._memory_cache)
            self.assertIn("session:user-1", cache._memory_cache)
            cache._memory_cache.clear()

    def test_chunk_ids_are_stable_and_source_specific(self):
        docs = [
            Document(page_content="same", metadata={"source": "a.md"}),
            Document(page_content="same", metadata={"source": "b.md"}),
        ]
        first = add_chunk_ids(docs)
        second_id = add_chunk_ids([
            Document(page_content="same", metadata={"source": "a.md"})
        ])[0].metadata["chunk_id"]

        self.assertNotEqual(first[0].metadata["chunk_id"], first[1].metadata["chunk_id"])
        self.assertEqual(first[0].metadata["chunk_id"], second_id)

    @patch("app.services.retriever.clear_retrieval_cache")
    @patch("app.services.hybrid_retriever.rebuild_bm25_index")
    @patch("app.services.indexer.get_vector_store")
    def test_indexing_rebuilds_bm25_and_clears_retrieval_cache(
        self, get_vector_store, rebuild_bm25_index, clear_retrieval_cache
    ):
        store = Mock()
        get_vector_store.return_value = store
        doc = Document(page_content="knowledge", metadata={"source": "a.md"})

        count = index_documents([doc])

        self.assertEqual(count, 1)
        store.add_documents.assert_called_once()
        call_kwargs = store.add_documents.call_args.kwargs
        self.assertEqual(call_kwargs["ids"], [
            store.add_documents.call_args.args[0][0].metadata["chunk_id"]
        ])
        rebuild_bm25_index.assert_called_once()
        clear_retrieval_cache.assert_called_once()


class RetrievalCompositionTests(unittest.TestCase):
    @patch("app.services.hybrid_retriever.rerank_documents")
    @patch("app.services.hybrid_retriever.rrf_fusion")
    @patch("app.services.hybrid_retriever.bm25_search")
    @patch("app.services.hybrid_retriever.vector_search")
    @patch("app.services.hybrid_retriever.generate_multi_queries", return_value=["q"])
    def test_enhanced_rrf_receives_vector_and_bm25_as_equal_ranked_lists(
        self, _queries, vector_search, bm25_search, rrf_fusion, rerank_documents
    ):
        vector_doc = Document(page_content="vector", metadata={"chunk_id": "vector-1"})
        bm25_doc = Document(page_content="bm25", metadata={"chunk_id": "bm25-1"})
        vector_search.return_value = [(vector_doc, 1.0)]
        bm25_search.return_value = [(bm25_doc, 2.0)]
        rrf_fusion.return_value = [(vector_doc, 1.0), (bm25_doc, 1.0)]
        rerank_documents.side_effect = lambda _query, docs, top_k: docs[:top_k]

        hybrid_retriever.hybrid_search("q", top_k=2, num_queries=1)

        ranked_lists = rrf_fusion.call_args.args[0]
        self.assertEqual(ranked_lists, [[(vector_doc, 1.0)], [(bm25_doc, 2.0)]])


class VectorCollectionTests(unittest.TestCase):
    @patch("app.db.vector_store.Chroma")
    @patch("app.db.vector_store.MiniMaxEmbeddings")
    def test_knowledge_and_memory_use_separate_collections(self, _embeddings, chroma):
        vector_store.clear_vector_store_cache()
        vector_store.get_vector_store()
        vector_store.get_memory_vector_store()

        self.assertEqual(
            chroma.call_args_list[0].kwargs["collection_name"],
            KNOWLEDGE_COLLECTION_NAME,
        )
        self.assertEqual(
            chroma.call_args_list[1].kwargs["collection_name"],
            MEMORY_COLLECTION_NAME,
        )
        self.assertNotEqual(KNOWLEDGE_COLLECTION_NAME, MEMORY_COLLECTION_NAME)
        vector_store.clear_vector_store_cache()

    @patch("app.db.vector_store.Chroma")
    @patch("app.db.vector_store.MiniMaxEmbeddings")
    def test_same_collection_reuses_one_client(self, _embeddings, chroma):
        vector_store.clear_vector_store_cache()
        first = vector_store.get_vector_store()
        second = vector_store.get_vector_store()

        self.assertIs(first, second)
        chroma.assert_called_once()
        vector_store.clear_vector_store_cache()


if __name__ == "__main__":
    unittest.main()
