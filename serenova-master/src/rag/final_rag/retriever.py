# 📦 Imports
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ AUTO PATH (correct and stable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# 🚨 Check DB exists
if not os.path.exists(DB_PATH):
    raise Exception(f"❌ chroma_db folder not found at: {DB_PATH}\nRun chroma_store.py first!")

print("✅ DB found at:", DB_PATH)

# 🔗 Load embedding model
print("🔗 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 📦 Load Chroma DB
print("📦 Loading Chroma DB...")
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

print("⚡ RETRIEVER READY")


# 🎯 Function (FINAL VERSION)
def get_relevant_rules(query):
    print("\n🚀 RUNNING RETRIEVER...")
    print("Query:", query)

    print("⏳ Searching in DB...")
    results = db.similarity_search(query, k=3)
    print("✅ Search completed")

    print("Results length:", len(results))

    if len(results) == 0:
        print("❌ No matching rules found!")
        return []

    print("\n🔍 TOP MATCHED RULES:\n")

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content.strip()}\n")

    return results   # ✅ REQUIRED FOR PIPELINE


# 🧪 Test (only when run directly)
if __name__ == "__main__":
    get_relevant_rules("SpO2 low and heart rate high")