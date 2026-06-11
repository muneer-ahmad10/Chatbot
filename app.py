
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)



st.set_page_config(page_title="PDF Chatbot")

st.title("📚 PDF RAG Chatbot")

st.write("Upload a PDF and ask questions.")



@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    llm = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    return embedding_model, tokenizer, llm




embedding_model, tokenizer, llm = load_models()



uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)



def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text

    return text


def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(
            text[i:i+chunk_size]
        )

    return chunks



if uploaded_file:
  pdf_text = extract_text(
        uploaded_file
    )

  chunks = create_chunks(
        pdf_text
    )

  chunk_embeddings = embedding_model.encode(
        chunks
    )
  



dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    np.array(chunk_embeddings)
    .astype("float32")
)



st.success(
    f"Indexed {len(chunks)} chunks."
)


question = st.text_input(
    "Ask a question"
)


def retrieve_context(question):

    q_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(q_embedding)
        .astype("float32"),
        k=3
    )

    retrieved = []

    for idx in indices[0]:

        if idx != -1:

            retrieved.append(
                chunks[idx]
            )

    return "\n".join(retrieved)





def answer_question(question):

    context = retrieve_context(
        question
    )

    prompt = f"""
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer








if question:

    with st.spinner(
        "Thinking..."
    ):

        answer = answer_question(
            question
        )

    st.subheader(
        "Answer"
    )

    st.write(answer)






