<div align="center">

# 📚 PDF RAG Chatbot

### Document Question Answering with Semantic Search

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/FLAN--T5-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/Vector_Search-FAISS-0467DF?style=for-the-badge)
![Colab](https://img.shields.io/badge/Developed_in-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

</div>

---

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions in natural language. The system retrieves relevant document chunks using semantic embeddings and FAISS, then uses Google's FLAN-T5 model to generate context-informed answers through a Streamlit chat interface.

> **Developed and tested in Google Colab**, with temporary public access provided through an ngrok tunnel.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture and Workflow](#-architecture-and-workflow)
- [Technology Stack](#️-technology-stack)
- [Retrieval and Generation Details](#-retrieval-and-generation-details)
- [Testing](#-testing)
- [Installation and Local Execution](#-installation-and-local-execution)
- [Running in Google Colab](#️-running-in-google-colab)
- [Current Limitations](#️-current-limitations)
- [Planned Improvements](#-planned-improvements)
- [Author](#-author)

---

## 🚀 Project Overview

Searching long PDF documents manually can be time-consuming. This project explores how retrieval and language generation can work together to make document-based question answering more convenient.

The application follows a RAG pipeline:

| Step | Stage |
|:--:|:--|
| **1** | Extract text from uploaded PDF documents. |
| **2** | Split the extracted text into overlapping chunks. |
| **3** | Convert each chunk into a semantic embedding. |
| **4** | Index the embeddings in FAISS for similarity search. |
| **5** | Retrieve the top three chunks relevant to a user's question. |
| **6** | Pass the retrieved context and question to FLAN-T5. |
| **7** | Display the generated answer and retrieved source excerpts in the chat interface. |

> The model is instructed to use the supplied document context and acknowledge when information is missing. This instruction helps guide responses but does not guarantee that every answer will be correct.

---

## ✨ Key Features

| Feature | Description |
|:--|:--|
| 📄 **Multiple PDF uploads** | Process multiple documents in one session. |
| 🔎 **PDF text extraction** | Extract readable text using PyPDF. |
| ✂️ **Overlapping chunking** | Use `RecursiveCharacterTextSplitter` with a 500-character chunk size and 100-character overlap. |
| 🧬 **Semantic embeddings** | Represent text and questions using `all-MiniLM-L6-v2`. |
| 🧭 **Vector similarity search** | Retrieve the top three candidate chunks with FAISS. |
| 🤖 **Context-informed generation** | Generate answers using `google/flan-t5-base`. |
| 💬 **Chat interface** | Ask follow-up questions through Streamlit. |
| 📌 **Source excerpts** | Display retrieved text and associated PDF filenames. |
| ⚡ **Cached model loading** | Use Streamlit's `@st.cache_resource` to avoid repeatedly loading the same models during app reruns. |
| ☁️ **Colab-based access** | Expose the Streamlit app temporarily through ngrok. |

---

## 🧠 Architecture and Workflow

```text
                  PDF Upload(s)
                       |
                       v
               PDF Text Extraction
                    (PyPDF)
                       |
                       v
             Recursive Text Splitting
              Chunk Size: 500
              Overlap: 100
                       |
                       v
             Sentence Embeddings
             all-MiniLM-L6-v2
                       |
                       v
               L2 Normalization
                       |
                       v
               FAISS Vector Index
                IndexFlatIP
                       |
                       |
User Question ----------+
       |
       v
Question Embedding + Normalization
       |
       v
Top-3 Similarity Search
       |
       v
Retrieved Document Context
       |
       +---- Conversation History
       |
       v
      FLAN-T5-base
       |
       v
    Generated Answer
       |
       v
Streamlit Chat + Source Excerpts
```

---

## 🛠️ Technology Stack

| Component | Technology |
|:--|:--|
| Programming language | Python |
| Interactive application | Streamlit |
| PDF parsing | PyPDF |
| Text splitting | LangChain Text Splitters |
| Embedding model | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector search | FAISS — `IndexFlatIP` |
| Similarity approach | Inner product on L2-normalized embeddings |
| Text generation | Hugging Face Transformers — `google/flan-t5-base` |
| Deep learning framework | PyTorch |
| Development environment | Google Colab |
| Temporary public access | pyngrok / ngrok |
| Secret management | Google Colab Secrets |

---

## 🔍 Retrieval and Generation Details

### 1️⃣ PDF text extraction

The application reads uploaded PDFs with `PdfReader` and extracts available text from their pages. The current implementation works with text-based PDFs; scanned documents may require an OCR step.

### 2️⃣ Text chunking

Long document text is split into smaller, overlapping chunks.

| Setting | Value |
|:--|:--|
| Chunk size | 500 characters |
| Chunk overlap | 100 characters |

Overlap helps preserve context when a relevant sentence or passage falls near a chunk boundary.

### 3️⃣ Embedding generation

The `all-MiniLM-L6-v2` Sentence Transformer converts each chunk into a numerical vector representing its semantic content. The user's question is embedded using the same model so it can be compared with the indexed document chunks.

### 4️⃣ FAISS similarity search

The application L2-normalizes document and query embeddings and uses `faiss.IndexFlatIP` to rank candidate chunks by inner product. For normalized vectors, this ranking corresponds to cosine-similarity ranking.

The three highest-ranked chunks are selected as context for answer generation.

### 5️⃣ Answer generation

The retrieved passages, question, and recent conversation history are formatted into a prompt and passed to `google/flan-t5-base` using the Hugging Face tokenizer and sequence-to-sequence model.

The prompt instructs the model to answer from the document context and acknowledge missing information. Retrieved passages are also displayed to help users inspect the evidence behind a response.

---

## 🧪 Testing

I tested the application using a fictional company-handbook PDF created specifically for RAG evaluation.

The test scenarios included:

- Retrieving stated facts, such as working hours and annual learning budget.
- Answering questions about remote-work policies and security reporting.
- Asking questions about information not specified in the document, such as the company's CEO.
- Inspecting retrieved text excerpts to check whether the relevant information was returned.

> These tests helped examine both retrieval and answer generation. They are functional checks, not a formal accuracy benchmark. No quantified retrieval or answer-quality score is claimed.

---

## 💻 Installation and Local Execution

### Prerequisites

- Python 3.10 or another Python version compatible with the selected package versions.
- Git.
- Sufficient memory and storage to download and run the embedding and language models.

### 1. Clone the repository

```bash
git clone https://github.com/muneer-ahmad10/Chatbot.git
cd Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

After adding a tested `requirements.txt` to the repository, run:

```bash
pip install -r requirements.txt
```

The core application dependencies include:

```text
streamlit
numpy
faiss-cpu
pypdf
sentence-transformers
transformers
torch
langchain-text-splitters
```

The exact versions should be pinned after verifying a clean installation. `pyngrok` is additionally needed for the Colab tunnel workflow.

### 4. Launch the application

Once `app.py` is saved in the repository root:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload one or more readable PDFs, and start asking questions.

> 📝 **Repository note:** The current implementation is embedded in `RAG_ChatBot.ipynb`. Commit the generated `app.py` and a tested `requirements.txt` before relying on these clone-and-run instructions.

---

## ☁️ Running in Google Colab

The development notebook installs the required libraries, writes the Streamlit application to `app.py`, starts the Streamlit server, and creates a temporary ngrok tunnel.

For authentication, store your ngrok token in Google Colab's Secrets panel as `NGROK_AUTH_TOKEN` and retrieve it in the notebook:

```python
from google.colab import userdata
from pyngrok import ngrok

ngrok_token = userdata.get("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(ngrok_token)

public_url = ngrok.connect(8501)
print(public_url)
```

> 🔐 Do not hardcode the token or commit it to GitHub. The public ngrok URL is temporary and may stop working when the Colab runtime ends or the tunnel is closed.

---

## ⚠️ Current Limitations

| Area | Limitation |
|:--|:--|
| **Answer reliability** | Prompt instructions do not eliminate hallucinations; generated answers still need verification. |
| **PDF coverage** | Text extraction does not automatically handle scanned or image-only documents. |
| **Source traceability** | The current implementation records PDF filenames and retrieved text but does not preserve page numbers for citations. |
| **Retrieval quality** | Selecting three chunks is a fixed setting and may not be optimal for every document or question. |
| **Evaluation** | The project has functional test cases but no formal retrieval benchmark or quantified answer-quality evaluation. |
| **Runtime and deployment** | Colab and ngrok provide a convenient demonstration environment, not a persistent production deployment. |
| **Reproducibility** | A committed `app.py` and tested, pinned dependencies are needed for a reliable clean setup. |

---

## 🔮 Planned Improvements

- [ ] Add page-level source citations.
- [ ] Build a repeatable evaluation dataset with answerable and unanswerable questions.
- [ ] Compare chunk sizes, overlap settings, and retrieval configurations.
- [ ] Improve handling of empty, corrupted, and scanned PDFs.
- [ ] Add a more robust strategy for detecting insufficient evidence.
- [ ] Cache document processing and indexes where appropriate.
- [ ] Add dependency version pins and a clean-install verification procedure.
- [ ] Deploy a persistent demo with suitable resource and security controls.

---

## 👨‍💻 Author

<div align="center">

**Muneer Ahmad Dar**

AI/ML &nbsp;|&nbsp; NLP &nbsp;|&nbsp; Retrieval-Augmented Generation &nbsp;|&nbsp; Semantic Search &nbsp;|&nbsp; PyTorch

[![GitHub](https://img.shields.io/badge/GitHub-muneer--ahmad10-181717?style=for-the-badge&logo=github)](https://github.com/muneer-ahmad10)
[![Repository](https://img.shields.io/badge/Project-Chatbot_Repository-blue?style=for-the-badge&logo=github)](https://github.com/muneer-ahmad10/Chatbot)

</div>

- GitHub: https://github.com/muneer-ahmad10
- Project repository: https://github.com/muneer-ahmad10/Chatbot
