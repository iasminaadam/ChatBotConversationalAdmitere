import os
import re
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Load the Database (Cale robustă bazată pe structura ta)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
PERSIST_DIRECTORY = os.path.join(root_dir, "database", "chroma_db")

print(f"📂 Verific baza de date la: {PERSIST_DIRECTORY}")

if not os.path.exists(PERSIST_DIRECTORY):
    print(f"❌ Error: Database not found at {PERSIST_DIRECTORY}!")
    exit()

print("🧠 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# 2. Setup LLM
llm = OllamaLLM(model="llama3.1", temperature=0.2)

# 3. Prompt Design
system_prompt = (
    "Ești un asistent virtual oficial al Universității Tehnice „Gheorghe Asachi” din Iași (TUIASI). "
    "Misiunea ta este să ajuți candidații cu informații precise despre admiterea 2026. "
    "Folosește fragmentele de mai jos (context) pentru a răspunde. Dacă nu știi, îndrumă-i către secretariat. "
    "Răspunde mereu în limba în care a fost pusă întrebarea (Română sau Engleză)."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Create the Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    vector_db.as_retriever(search_kwargs={"k": 5}), 
    combine_docs_chain
)

def get_source_from_text(text):
    match = re.search(r"SOURCE:\s*(https?://\S+)", text)
    return match.group(1) if match else None

# 5. Interactive Chat
print("\n" + "="*50)
print("🤖 TUIASI Chatbot (v1.2 - Classic) is online!")
print("Type 'exit' to stop.")
print("="*50)

while True:
    user_input = input("\nTu: ")
    if user_input.lower() in ['exit', 'quit', 'iesire']:
        break
    
    if not user_input.strip():
        continue
        
    # Process request
    response = rag_chain.invoke({"input": user_input})
    
    # --- SECȚIUNE NOUĂ: Printare Fragmente Consultate ---
    print("\n" + "-"*30)
    print("📄 DOCUMENTE EXTRASE DIN BAZA DE DATE:")
    for i, doc in enumerate(response['context']):
        print(f"\n[Fragment {i+1}]:")
        print(f"{doc.page_content}")
        print("-" * 20)
    print("-"*30)
    # ---------------------------------------------------

    print(f"\nAsistent: {response['answer']}")
    
    # Extragem sursele
    sources = []
    for doc in response['context']:
        source = get_source_from_text(doc.page_content)
        if source:
            sources.append(source)
    
    unique_sources = list(set(sources))
    if unique_sources:
        print("\n📚 Surse (link-uri):")
        for s in unique_sources:
            print(f"- {s}")