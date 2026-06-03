import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ✅ AUTO PATH (CORRECT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")


def load_documents():
    docs = []

    if not os.path.exists(DATA_PATH):
        raise Exception(f"❌ DATA folder not found at: {DATA_PATH}")

    for file in os.listdir(DATA_PATH):
        print("Loading:", file)
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
            docs.extend(loader.load())

    print("Total docs:", len(docs))
    return docs


def create_chroma_db():
    print("🚀 STARTING DB CREATION...")

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(documents)

    print("Chunks:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Creating Chroma DB...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"✅ DB CREATED SUCCESSFULLY at: {DB_PATH}")


if __name__ == "__main__":
    create_chroma_db()