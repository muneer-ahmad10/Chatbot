# 📚 PDF RAG Chatbot

> A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it in natural language — powered by semantic search and a local open-source LLM.

🔗 **Live Demo:** [handbook-marry-hazy.ngrok-free.dev](https://handbook-marry-hazy.ngrok-free.dev/)

---

## 🎯 Overview

Most chatbots either hallucinate answers or can't access your private documents. This project solves that by combining:

- **Semantic retrieval** to find the most relevant sections of a document
- **A local language model (FLAN-T5)** to generate grounded answers using *only* retrieved context

The result: a chatbot that answers questions strictly based on the content of the PDF you upload — no external knowledge, no hallucinated facts.

---

## 🧠 How It Works (RAG Pipeline)

```
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Chunking (500-character chunks)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
FAISS Vector Index (IndexFlatL2)
    ↓
User Question → Embed → Similarity Search (top-3 chunks)
    ↓
Context + Question → Prompt → FLAN-T5-base
    ↓
Generated Answer
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI / App Framework | Streamlit |
| PDF Parsing | PyPDF |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (`IndexFlatL2`) |
| LLM | HuggingFace `google/flan-t5-base` (Seq2Seq) |
| Deployment | Streamlit + ngrok tunnel |

---

## 🚀 Features

- 📄 **Upload any PDF** and instantly index its content
- 🔍 **Semantic search** — retrieves meaning-relevant chunks, not just keyword matches
- 🤖 **Context-grounded answers** — LLM only responds based on retrieved document context
- ⚡ **Cached model loading** (`@st.cache_resource`) for fast repeated queries
- 🌐 **Live deployed demo** accessible via ngrok

---

## 📂 Project Structure

```
Chatbot/
├── app.py              # Main Streamlit RAG application (production)
├── RAG_ChatBot.ipynb   # Development notebook — RAG pipeline experimentation
├── Chatbot.ipynb       # Early experimentation / practice notebook
└── README.md
```

> `app.py` contains the finalized, deployable version of the RAG pipeline developed and tested in `RAG_ChatBot.ipynb`. The other notebook (`Chatbot.ipynb`) was used for early practice and isn't part of the core app.

---

## 🛠️ Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/muneer-ahmad10/Chatbot.git
cd Chatbot

# 2. Install dependencies
pip install streamlit pypdf sentence-transformers faiss-cpu transformers torch

# 3. Run the app
streamlit run app.py
```

Then open the local URL shown in your terminal, upload a PDF, and start asking questions.

---

## 🔮 Future Improvements

- [ ] Support for multi-document chat / knowledge base
- [ ] Swap FLAN-T5 for a larger instruction-tuned LLM (e.g., Mistral, Llama)
- [ ] Persistent vector store (instead of in-memory FAISS)
- [ ] Chat history with multi-turn context
- [ ] Source citation — show which page/chunk an answer came from
- [ ] Streaming responses for better UX

---

## 👨‍💻 Author

**Muneer Ahmad Dar** — AI Engineer  
Focus: NLP · LLMs · RAG Systems · Semantic Search

[![GitHub](https://img.shields.io/badge/GitHub-muneer--ahmad10-black?style=flat&logo=github)](https://github.com/muneer-ahmad10)
