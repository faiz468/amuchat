# AMU Admission Query Chatbot

The **AMU Admission Query Chatbot** is a conversational AI tool designed to assist students with queries related to the admission procedures at **Aligarh Muslim University (AMU)**. This chatbot leverages **RAG (Retrieval-Augmented Generation)** architecture to provide accurate and context-aware answers based on official admission documents.
![Image]([https://github.com/faiz-mubeen/Multi-Lingual-Legal-Question-Answering-System/blob/main/data/perf_comparison.png](https://github.com/faiz468/amuchat/blob/main/static/demonstration.png))
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
- **[Sources](https://github.com/faiz468/amuchat/tree/main/data)**:
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
## Installation and Setup Guide

Follow the steps below to set up and run the **AMU Admission Query Chatbot** on your local machine.

---

### Prerequisites
Ensure that you have the following installed:
1. **Python**: Version 3.8 or higher.
2. **Git**: To clone the repository.
3. **NVIDIA API Key**: Required for the chatbot to function.

---

### Steps to Install and Run

1. Clone the Repository
Use the following command to clone the repository to your local machine:
```bash
git clone https://github.com/faiz468/amuchat.git
```

2. Navigate to the Project Directory
Change your working directory to the project folder:
```bash
cd amuchat
```

3. Create a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to keep dependencies isolated:
```bash
python -m venv venv
venv\Scripts\activate
```

4. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

5. Configure Environment Variables
Create a .env file in the project root directory and add your NVIDIA API Key:
```plaintext
NVIDIA_API_KEY=your_api_key
```

6. Run the Application
Start the Streamlit app using the following command:
```bash
streamlit run app.py
```
