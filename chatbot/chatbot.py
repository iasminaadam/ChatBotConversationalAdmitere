import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 👇 These come from the new langchain-classic package
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Load the Database
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
PERSIST_DIRECTORY = os.path.join(parent_dir, "database/db_admitere")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# 2. Setup LLM (Make sure Ollama app is running!)
llm = OllamaLLM(model="llama3.1")

# 3. Prompt Design
system_prompt = (
    "Ești un asistent virtual oficial al Universității Tehnice „Gheorghe Asachi” din Iași (TUIASI). "
    "Misiunea ta este să ajuți candidații cu informații precise despre admitere. "
    "Folosește fragmentele de mai jos pentru a răspunde. Dacă nu știi, spune că nu știi și îndrumă utilizatorul către secretariat."
    "Răspunde mereu în limba în care a fost pusă întrebarea (Română sau Engleză)."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Create the Chain (Modern LangChain 1.x Style)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)

# 5. Interactive Chat
print("\n🤖 TUIASI Chatbot (v1.2) is online! (Type 'exit' to stop)")
while True:
    user_input = input("\nTu: ")
    if user_input.lower() in ['exit', 'quit', 'iesire']:
        break
        
    # In LangChain 1.x, we use 'input' for queries and 'answer' for results
    response = rag_chain.invoke({"input": user_input})
    
    print(f"\nAsistent: {response['answer']}")
    
    # Show sources found in the 'context' metadata
    print("\n📚 Surse consultate:")
    sources = set([doc.metadata.get('url') for doc in response['context']])
    for s in sources:
        print(f"- {s}")