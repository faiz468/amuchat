import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Initialize title for Streamlit app
st.title("AMU Admission Assistant")

# Creating a sidebar with a title and brief description about LegalEase
with st.sidebar:
    st.image("static/amu.png", width=32)
    st.title("AMU Admission")
    st.subheader("About")
    st.markdown(
        """
        AMU Admission Assistant answers student queries regarding admission to the university.
        You can ask anything about the admission procedure of the university.
        """
    )

# Adding a divider in the UI
st.divider()

# Set up the LLM
llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a question answering system for Aligarh Muslim University who will answer students' queries regarding admission to the university. Please answer the following question accurately based solely on the provided context. Follow these instructions carefully:

    1. **Rely Only on the Context:** Use only the information provided in the context section to answer the question. Do not use any outside knowledge or make assumptions.
    2. **Respond with Precision:** Provide a clear and concise answer. If a question requires specific details, include them as precisely as they appear in the context.
    3. **Indicate Missing Information:** If the context does not contain the information needed to answer the question, respond with: "Answer is not available in the provided context." Avoid adding guesses or unrelated information.
    4. **Maintain Neutrality:** Avoid adding opinions or additional explanations unless specifically requested in the question.

    **Context:**
    {context}
    **Question:**
    {input}
    """
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./data")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting

        # Check if the FAISS index already exists locally
        if os.path.exists("faiss_index"):
            st.session_state.vectors = FAISS.load_local("faiss_index", st.session_state.embeddings, allow_dangerous_deserialization=True)
            st.write("Loaded FAISS Vector Store from local storage.")
        else:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.session_state.vectors.save_local("faiss_index")  # Save the index for later use
            st.write("FAISS Vector Store DB created and saved locally.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input for query
query = st.text_input("Ask Your Question...", placeholder="Type your question here...")

# Button to create embeddings
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("FAISS Vector Store DB is ready using NVIDIA Embedding.")
    
if "vectors" not in st.session_state:
    vector_embedding()

# Initialize response variable
response = None

# Process the query
if query:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': query})
    response_time = time.process_time() - start

    st.session_state.history.append({"origin": "user", "message": query})
    st.session_state.history.append({"origin": "ai", "message": response['answer']})

    st.write(f"Response time: {response_time:.2f} seconds")

# Display chat history with Streamlit elements only
st.write("### Chat History")

for chat in st.session_state.history:
    if chat["origin"] == "user":
        # Display user message on the right with light green background
        with st.container():
            cols = st.columns([2, 10])
            with cols[1]:
                st.image("static/profile.png", width=32)
                st.markdown(
                    f"<div style='background-color: #d4edda; padding: 10px; border-radius: 10px; margin: 5px;'>"
                    f"{chat['message']}</div>",
                    unsafe_allow_html=True
                )
    else:
        # Display AI message on the left with white background
        with st.container():
            cols = st.columns([10, 2])
            with cols[0]:
                st.image("static/amu.png", width=32)
                st.markdown(
                    f"<div style='background-color: #ffffff; padding: 10px; border-radius: 10px; margin: 5px;'>"
                    f"{chat['message']}</div>",
                    unsafe_allow_html=True
                )

# Expander for document similarity search if response is available
if response:
    with st.expander("Document Similarity Search"):
        for doc in response.get("context", []):
            st.write(doc.page_content)
            st.write("--------------------------------")

# import time

# Display AI message with typing animation
# def display_typing_animation(text):
#     typed_text = ""
#     for char in text:
#         typed_text += char
#         st.markdown(f"<div style='background-color: #ffffff; padding: 10px; border-radius: 10px; margin: 5px;'>{typed_text}</div>", unsafe_allow_html=True)
#         time.sleep(0.03)  # Adjust typing speed here

# # Inside your response handling section
# if response:
#     st.session_state.history.append({"origin": "user", "message": query})
#     st.session_state.history.append({"origin": "ai", "message": response['answer']})
    
#     # Display typing animation for AI response
#     display_typing_animation(response['answer'])

#     st.write(f"Response time: {response_time:.2f} seconds")
