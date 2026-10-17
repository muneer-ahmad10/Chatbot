
from text_processing import create_chunks

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
