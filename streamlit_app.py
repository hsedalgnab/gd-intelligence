import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(
    page_title="GD Intelligence",
    page_icon="⚖️",
    layout="centered",
    menu_items={
        "About": "GD Intelligence - AI-powered legal document intelligence for Gibson, Dunn & Crutcher LLP"
    }
)

st.title("GD Intelligence ⚖️")
st.caption("AI-powered legal intelligence for Gibson, Dunn & Crutcher LLP.")
st.caption("AI can make mistakes. Always verify with original case documents.")

# Load the FAISS index and the HuggingFace embeddings model
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Initialize the LLM using a custom OAI-compatible endpoint
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL")
)

# Keep track of the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Accept user input
if question := st.chat_input("Ask a question about Gibson Dunn's cases..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Retrieve relevant chunks and build prompt
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', '?')}\n{doc.page_content}"
        for doc in relevant_docs
    ])
    prompt = f"You are a legal assistant for Gibson, Dunn & Crutcher LLP. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"

    # Stream the response from the LLM
    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.content for chunk in llm.stream(prompt)
        )

    st.session_state.messages.append({"role": "assistant", "content": response})