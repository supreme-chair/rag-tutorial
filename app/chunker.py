"""Chunk documents into smaller pieces."""
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import PROCESSED_DATA_DIR, CHUNK_MAX_CHARS, CHUNK_OVERLAP


def chunk_documents(
    input_file: Path = None,
    output_file: Path = None,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Split documents into chunks by paragraphs with size limit."""
    if input_file is None:
        input_file = PROCESSED_DATA_DIR / "documents.jsonl"
    if output_file is None:
        output_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    
    chunks = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            name = doc["name"]
            text = doc["text"]
            source = doc.get("source_file", "unknown")
            
            # Split by paragraphs (double newline)
            paragraphs = text.split("\n\n")
            current_chunk = ""
            chunk_idx = 0
            
            for para in paragraphs:
                # If a single paragraph is too long, split by sentences
                if len(para) > max_chars:
                    # Simple sentence splitting (., !, ?)
                    sentences = para.replace("! ", ". ").replace("? ", ". ").split(". ")
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 2 <= max_chars:
                            if current_chunk:
                                current_chunk += ". " + sent
                            else:
                                current_chunk = sent
                        else:
                            if current_chunk:
                                chunks.append({
                                    "chunk_id": f"{doc_id}_chunk_{chunk_idx:03d}",
                                    "doc_id": doc_id,
                                    "name": name,
                                    "text": current_chunk.strip(),
                                    "source": source,
                                })
                                chunk_idx += 1
                            current_chunk = sent
                else:
                    # Normal paragraph
                    if len(current_chunk) + len(para) + 2 <= max_chars:
                        if current_chunk:
                            current_chunk += "\n\n" + para
                        else:
                            current_chunk = para
                    else:
                        if current_chunk:
                            chunks.append({
                                "chunk_id": f"{doc_id}_chunk_{chunk_idx:03d}",
                                "doc_id": doc_id,
                                "name": name,
                                "text": current_chunk.strip(),
                                "source": source,
                            })
                            chunk_idx += 1
                            # Keep overlap
                            if overlap > 0 and len(current_chunk) > overlap:
                                current_chunk = current_chunk[-overlap:] + "\n\n" + para
                            else:
                                current_chunk = para
                        else:
                            current_chunk = para
            
            # Don't forget the last chunk
            if current_chunk:
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "name": name,
                    "text": current_chunk.strip(),
                    "source": source,
                })
    
    # Write chunks to file
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"Created {len(chunks)} chunks from documents")
    return chunks


if __name__ == "__main__":
    chunk_documents()