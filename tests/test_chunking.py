import pytest
import json
import tempfile
from pathlib import Path

from app.chunker import chunk_documents


def test_chunking_creates_chunks():
    """Test that chunking creates chunks from documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_file = tmp / "documents.jsonl"
        output_file = tmp / "chunks.jsonl"
        
        # Create test document
        with open(input_file, "w", encoding="utf-8") as f:
            doc = {
                "doc_id": "doc_0001",
                "name": "Test Doc",
                "text": "First paragraph.\n\nSecond paragraph with more text.",
                "source_file": "test"
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        # Run chunking
        chunks = chunk_documents(input_file, output_file, max_chars=1000, overlap=50)
        
        assert len(chunks) > 0
        assert output_file.exists()
        
        # Check chunk structure
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert "text" in chunk
            assert chunk["doc_id"] == "doc_0001"


def test_chunking_respects_max_chars():
    """Test that chunks don't exceed max_chars."""
    long_text = "This is a very long sentence. " * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_file = tmp / "documents.jsonl"
        output_file = tmp / "chunks.jsonl"
        
        with open(input_file, "w", encoding="utf-8") as f:
            doc = {
                "doc_id": "doc_0001",
                "name": "Long Doc",
                "text": long_text,
                "source_file": "test"
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        chunks = chunk_documents(input_file, output_file, max_chars=500, overlap=50)
        
        for chunk in chunks:
            assert len(chunk["text"]) <= 500 + 50  # Allow small overlap overhead


def test_chunking_preserves_metadata():
    """Test that chunking preserves document metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_file = tmp / "documents.jsonl"
        output_file = tmp / "chunks.jsonl"
        
        with open(input_file, "w", encoding="utf-8") as f:
            doc = {
                "doc_id": "doc_0001",
                "name": "Test Document",
                "text": "Some text for testing.",
                "source_file": "test_source"
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        chunks = chunk_documents(input_file, output_file, max_chars=1000, overlap=50)
        
        for chunk in chunks:
            assert chunk["doc_id"] == "doc_0001"
            assert chunk["name"] == "Test Document"
            assert chunk["source"] == "test_source"