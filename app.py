from text_processing import create_chunks
import streamlit as st
import numpy as np
import faiss
import hashlib
import io
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="PDF Chatbot")
st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions.")

# Load Models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedding_model, tokenizer, llm

embedding_model, tokenizer, llm = load_models()

#Helper Functions
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# def create_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )
#     return splitter.split_text(text)




def answer_question(question, index, all_chunks):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )
    query_embedding = np.asarray(
        query_embedding, dtype=np.float32
    )
    faiss.normalize_L2(query_embedding)

    k = min(3, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_sources = []

    for idx in indices[0]:
        if 0 <= idx < len(all_chunks):
            retrieved_chunks.append(all_chunks[idx]["text"])
            retrieved_sources.append(all_chunks[idx]["source"])

    context = "\n\n".join(retrieved_chunks)

    history = ""
    for msg in st.session_state.messages[-4:]:
        history += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You answer questions using the document context below.

Rules:
1. Answer the question directly and in a complete sentence.
2. Use only information explicitly stated in the context.
3. If the answer is not provided, say:
   "The document does not specify this information."
4. Do not copy an incomplete phrase from the context as your answer.
5. Do not guess or invent names, facts, or details.

Conversation history:
{history}

Document context:
{context}

Question: {question}
Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, retrieved_chunks, retrieved_sources, scores



with st.sidebar:

    st.header("Project Info")

    st.write(
        """
        PDF RAG Chatbot

        - Sentence Transformers
        - FAISS
        - FLAN-T5
        - Streamlit
        """
    )



#Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar / Upload Section
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

#Process PDFs and Build FAISS Index if files are present
index = None
chunks = []


# Process uploaded PDFs
if uploaded_files:
    file_hashes = [
        hashlib.sha256(pdf.getvalue()).hexdigest()
        for pdf in uploaded_files
    ]
    files_id = hashlib.sha256(
        "".join(file_hashes).encode()
    ).hexdigest()

    if st.session_state.get("files_id") != files_id:
        st.session_state["files_id"] = files_id
        st.session_state["chunks"] = []
        st.session_state["all_chunks"] = []
        st.session_state["index"] = None
        st.session_state["messages"] = []

        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        try:
            for pdf in uploaded_files:
                pdf_text = extract_text(io.BytesIO(pdf.getvalue()))

                if not pdf_text.strip():
                    st.warning(
                        f"No extractable text found in {pdf.name}."
                    )
                    continue

                pdf_chunks = splitter.split_text(pdf_text)

                for chunk in pdf_chunks:
                    all_chunks.append({
                        "text": chunk,
                        "source": pdf.name
                    })

            if all_chunks:
                texts = [item["text"] for item in all_chunks]

                embeddings = embedding_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings = np.asarray(
                    embeddings, dtype=np.float32
                )
                faiss.normalize_L2(embeddings)

                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)

                st.session_state["all_chunks"] = all_chunks
                st.session_state["chunks"] = texts
                st.session_state["index"] = index
            else:
                st.warning(
                    "No readable text was found in the uploaded PDFs."
                )

        except Exception as error:
            st.session_state["index"] = None
            st.session_state["chunks"] = []
            st.session_state["all_chunks"] = []
            st.error(f"PDF processing failed: {error}")

# Use the processed index
index = st.session_state.get("index")
all_chunks = st.session_state.get("all_chunks", [])

if index is not None and all_chunks:
    st.success(f"Indexed {len(all_chunks)} chunks.")

    # Display previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            for i, source in enumerate(message.get("sources", [])):
                with st.expander(
                    f"Source {i + 1} — {source['name']}"
                ):
                    st.write(source["text"])

    # Handle a new question
    prompt = st.chat_input("Ask a question about your PDFs")

    if prompt and prompt.strip():
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Searching documents and generating an answer..."):
            try:
                answer, retrieved_chunks, retrieved_sources, scores = (
                    answer_question(prompt, index, all_chunks)
                )

                source_details = [
                    {"name": name, "text": chunk}
                    for name, chunk in zip(
                        retrieved_sources, retrieved_chunks
                    )
                ]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_details
                })

                st.rerun()

            except Exception as error:
                st.error(f"Could not answer the question: {error}")

else:
    st.info("Upload one or more readable PDFs to begin.")


