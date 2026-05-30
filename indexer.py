import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS_DIR = "documents"
INDEX_DIR = "faiss_index"

# Load all PDF files from the documents directory
def load_documents():
    docs = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DOCUMENTS_DIR, filename)
            print(f"Loading document: {filename}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs


# Split, embed, and save the FAISS index
def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunk = splitter.split_documents(docs)
    print(f"Total chunk: {len(chunk)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunk, embeddings)
    db.save_local(INDEX_DIR)
    print(f"FAISS index saved to {INDEX_DIR}")


if __name__ == "__main__":
    docs = load_documents()
    print(f"Total pages loaded: {len(docs)}")
    build_index(docs)
    print("Indexing complete.")



