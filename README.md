# AMU Admission Query Chatbot

## Overview
The **AMU Admission Query Chatbot** is a conversational AI tool designed to assist students with queries related to the admission procedures at **Aligarh Muslim University (AMU)**. This chatbot leverages **RAG (Retrieval-Augmented Generation)** architecture to provide accurate and context-aware answers based on official admission documents.

## Features
- **Comprehensive Coverage**: Answers questions about admissions, courses, and university policies.
- **Document Integration**: Utilizes official guides such as:
  - University admission guide
  - Program-specific guides (PG, PhD, NRI admissions, etc.)
  - Information policy and rulebook
- **Streamlit-based UI**: A user-friendly interface for seamless interactions.
- **Advanced Retrieval**: Uses FAISS indexing for efficient document search.
- **NVIDIA AI Integration**: Embeds cutting-edge language models like `nvidia/llama-3.1-nemotron-70b-instruct`.

---

## Key Components
### Dataset
- **Sources**:
  - Official guides and rulebooks provided by AMU.
  - Specific documents for various admission programs.
- **Processing**:
  - Documents are split into chunks using a recursive text splitter.
  - Embedded into a FAISS vector store for retrieval.

### Architecture
- **RAG (Retrieval-Augmented Generation)**:
  - Combines retrieval-based techniques with generative AI for accurate responses.
  - Embedding creation using **NVIDIAEmbeddings**.
  - FAISS for efficient vector-based document retrieval.
- **Prompt Engineering**:
  - Guides the chatbot to generate precise and context-restricted answers.

### Technology Stack
- **Programming Language**: Python
- **Frameworks**:
  - Streamlit for the web application interface.
  - LangChain for managing LLM chains and retrieval workflows.
- **Libraries**:
  - FAISS for vector search.
  - PyPDFLoader for document ingestion and processing.

---

