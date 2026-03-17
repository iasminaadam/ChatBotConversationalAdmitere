#build_database.py
import json
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the processed data
print("📂 Loading processed_knowledge.json...")
try:
    with open("processed_knowledge.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print("❌ Error: processed_knowledge.json not found. Run the extraction script first!")
    exit()

# --- PASUL 1: Pregătire documente ---
documents = []
for item in raw_data:
    metadata = item["metadata"]
    
    # Extragem tag-urile vechi și le transformăm în string
    tags_list = metadata.get("tags", [])
    if tags_list:
        metadata["tags_str"] = ", ".join(tags_list)
    else:
        metadata["tags_str"] = "general"
    
    # Ștergem lista originală ca să nu mai dea eroare în ChromaDB
    if "tags" in metadata:
        del metadata["tags"]
    
    doc = Document(page_content=item["content"], metadata=metadata)
    documents.append(doc)

# --- PASUL 2: Split în chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# Injectăm tag-urile în text folosind cheia cea nouă 'tags_str'
for chunk in chunks:
    t_str = chunk.metadata.get("tags_str", "general")
    if t_str != "general":
        # Adăugăm tag-urile la începutul textului pentru a ajuta motorul de căutare
        tag_context = f"[Categorii: {t_str}] "
        chunk.page_content = tag_context + chunk.page_content

print(f"✂️ Split {len(documents)} documente în {len(chunks)} chunks.")

# 3. Initialize Multilingual Embeddings
print("🧠 Initializing Multilingual Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 4. Create and Persist the Vector Database
PERSIST_DIRECTORY = "db_admitere"

# Clean up old database if it exists to avoid mixing old data without tags
if os.path.exists(PERSIST_DIRECTORY):
    print("🧹 Cleaning up old database version...")
    import shutil
    shutil.rmtree(PERSIST_DIRECTORY)

print(f"🚀 Building Vector Database in '{PERSIST_DIRECTORY}'...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print("✅ Database built successfully with metadata tags!")

# 5. Quick Test Search with Tag Filtering
# This is how you would search specifically for 'licenta' content
query = "Care sunt actele necesare?"
print(f"\n🔍 Testing search for: '{query}' with filter [licenta]")

# similarity_search supports metadata filtering!
docs = vector_db.similarity_search(
    query, 
    k=3, 
    filter={"tags_str": {"$contains": "licenta"}} # This looks for 'licenta' in our string
)

if not docs: # Fallback to normal search if filter is too strict
    docs = vector_db.similarity_search(query, k=3)

print("\n🔎 Top results:")
for i, doc in enumerate(docs):
    print(f"\n[{i+1}] Source: {doc.metadata.get('url')}")
    print(f"Tags: {doc.metadata.get('tags')}")
    print(f"Content: {doc.page_content[:150]}...")