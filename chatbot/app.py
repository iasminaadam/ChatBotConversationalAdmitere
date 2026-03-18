#app.py
import os
import re
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="TUIASI Admission Bot", page_icon="🎓")
st.title("🎓 TUIASI Admission Assistant 2026")
st.markdown("Întreabă-mă orice despre admitere, taxe sau documente!")

# --- LOGICĂ BACKEND (Cache pentru a nu reîncărca modelul la fiecare click) ---
@st.cache_resource
def init_rag_chain():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    PERSIST_DIRECTORY = os.path.join(root_dir, "database", "chroma_db")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    llm = OllamaLLM(model="llama3.1", temperature=0.2)

    system_prompt = (
        "Ești un asistent virtual oficial al Universității Tehnice „Gheorghe Asachi” din Iași (TUIASI). "
        "Misiunea ta este să ajuți candidații cu informații precise despre admiterea 2026. "
        "Folosește fragmentele de mai jos (context) pentru a răspunde. Dacă nu știi, îndrumă-i către secretariat. "
        "Răspunde mereu în limba în care a fost pusă întrebarea.\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 5}), combine_docs_chain)

rag_chain = init_rag_chain()

def get_source_from_text(text):
    match = re.search(r"SOURCE:\s*(https?://\S+)", text)
    return match.group(1) if match else None

# --- INTERFAȚA DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afișăm istoricul mesajelor
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Câmpul de input pentru utilizator
if user_input := st.chat_input("Cu ce te pot ajuta?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Caut în baza de date..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response['answer']
            
            # Afișăm răspunsul
            st.markdown(answer)
            
            # --- Sidebar sau Expander pentru Documente Consultate ---
            with st.expander("📄 Vezi documentele consultate (Chunks)"):
                for i, doc in enumerate(response['context']):
                    st.info(f"**Fragment {i+1}:**\n\n{doc.page_content}")
            
            # Afișăm sursele
            sources = list(set([get_source_from_text(d.page_content) for d in response['context'] if get_source_from_text(d.page_content)]))
            if sources:
                st.markdown("**Surse oficiale:**")
                for s in sources:
                    st.markdown(f"- [{s}]({s})")

    st.session_state.messages.append({"role": "assistant", "content": answer})