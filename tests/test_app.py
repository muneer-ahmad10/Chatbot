
from text_processing import create_chunks
from retrieval import retrieve_chunks

import faiss
import numpy as np



def test_create_chunks_returns_text_chunks():
    text = "Machine learning helps computers learn. " * 100

    chunks = create_chunks(text)

    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_create_chunks_handles_short_text():
    text = "This is a short test document."

    chunks = create_chunks(text)

    assert len(chunks) >= 1
    assert any(text in chunk for chunk in chunks)
    

def test_retrieves_most_relevant_chunk_first():
    chunks = [
        {"text": "The company's CEO is Sarah Khan.", "source": "company.pdf"},
        {"text": "The company sells office furniture.", "source": "company.pdf"},
        {"text": "The weather is sunny today.", "source": "weather.pdf"},
    ]

    # Small, deterministic embeddings for testing FAISS ranking.
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    query = np.array([[0.99, 0.01, 0.0]], dtype=np.float32)

    retrieved, sources, scores = retrieve_chunks(
        query, index, chunks, top_k=2
    )

    assert retrieved[0] == "The company's CEO is Sarah Khan."
    assert sources[0] == "company.pdf"
    assert len(retrieved) == 2
    assert scores.shape == (1, 2)


def test_retrieval_handles_empty_index():
    index = faiss.IndexFlatIP(3)

    retrieved, sources, scores = retrieve_chunks(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        index,
        [],
    )

    assert retrieved == []
    assert sources == []
    assert scores.shape == (1, 0)


def test_retrieval_respects_top_k():
    chunks = [
        {"text": "Chunk A", "source": "a.pdf"},
        {"text": "Chunk B", "source": "b.pdf"},
        {"text": "Chunk C", "source": "c.pdf"},
    ]

    vectors = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    retrieved, sources, scores = retrieve_chunks(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        index,
        chunks,
        top_k=1,
    )

    assert retrieved == ["Chunk A"]
    assert sources == ["a.pdf"]
    assert scores.shape == (1, 1)

