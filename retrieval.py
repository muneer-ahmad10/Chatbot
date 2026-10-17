
import numpy as np


def retrieve_chunks(query_embedding, index, all_chunks, top_k=3):
    """Retrieve the most similar document chunks from a FAISS index."""

    if index is None or not all_chunks or index.ntotal == 0:
        return [], [], np.empty((1, 0), dtype=np.float32)

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError("Expected one query embedding.")

    # Normalize the query for cosine-similarity search.
    norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    if norm[0, 0] > 0:
        query_embedding = query_embedding / norm

    k = min(max(1, top_k), index.ntotal)
    scores, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_sources = []

    for idx in indices[0]:
        if 0 <= idx < len(all_chunks):
            retrieved_chunks.append(all_chunks[idx]["text"])
            retrieved_sources.append(all_chunks[idx]["source"])

    return retrieved_chunks, retrieved_sources, scores
