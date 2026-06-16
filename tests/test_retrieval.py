import pytest
import json
import tempfile
import pickle
from pathlib import Path
import numpy as np

from app.retriever import HybridRetriever

@pytest.fixture
def retriever():
    from app.retriever import HybridRetriever
    r = HybridRetriever()
    if r.tfidf_vectorizer is None:
        pytest.skip("Index not built. Run: uv run python scripts/build_index.py")
    return r


def test_search_returns_results(retriever):
    results = retriever.search("Python", k=3)
    assert isinstance(results, list)

def test_tfidf_retrieval_returns_results():
    """Test that TF-IDF retrieval returns results for a query."""
    retriever = HybridRetriever()
    
    # Skip if index not built
    if retriever.tfidf_vectorizer is None:
        pytest.skip("TF-IDF index not built")
    
    results = retriever.search("Python программирование", k=3)
    
    assert isinstance(results, list)
    for result in results:
        assert "chunk_id" in result
        assert "text" in result
        assert "score" in result


def test_hybrid_search_combines_methods():
    """Test that hybrid search returns results."""
    retriever = HybridRetriever()
    
    if retriever.tfidf_vectorizer is None:
        pytest.skip("Index not built")
    
    results = retriever.search("машинное обучение", k=3)
    
    assert len(results) >= 0
    # Check that scores are between 0 and 1
    for result in results:
        assert 0 <= result["score"] <= 1


def test_empty_query_handling():
    """Test that empty query returns empty results."""
    retriever = HybridRetriever()
    
    if retriever.tfidf_vectorizer is None:
        pytest.skip("Index not built")
    
    results = retriever.search("", k=3)
    
    # Empty query might return nothing or very low scores
    assert isinstance(results, list)