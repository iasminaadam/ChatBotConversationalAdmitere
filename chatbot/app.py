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

# --- LOGICĂ BACKEND ---
@st.cache_resource
def init_rag_chain():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    PERSIST_DIRECTORY = os.path.join(root_dir, "database", "chroma_db")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    # Am adăugat un timeout de 120 secunde pentru a preveni blocajele pe PC-uri mai lente
    llm = OllamaLLM(model="llama3.1", temperature=0.2, timeout=120)

    # Prompt actualizat care include placeholder-ul pentru ISTORIC
    system_prompt = (
        "Ești un asistent virtual oficial al Universității Tehnice „Gheorghe Asachi” din Iași (TUIASI). "
        "Misiunea ta este să ajuți candidații cu informații precise despre admiterea 2026.\n\n"
        "REGULI:\n"
        "1. Folosește CONTEXTUL de mai jos pentru fapte și date.\n"
        "2. Folosește ISTORICUL CONVERSAȚIEI pentru a înțelege contextul întrebărilor scurte sau de continuare.\n"
        "3. Dacă întrebarea este GENERALĂ, caută informații care se aplică întregii universități.\n"
        "4. Nu generaliza o regulă de la o singură facultate la toată universitatea.\n"
        "5. Dacă nu știi, îndrumă utilizatorul către https://www.tuiasi.ro/admitere/.\n"
        "6. Răspunde în limba întrebării.\n\n"
        "ISTORIC CONVERSAȚIE:\n{history}\n\n"
        "CONTEXT DIN BAZA DE DATE:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 10}), combine_docs_chain)

rag_chain = init_rag_chain()

def get_source_from_text(text):
    match = re.search(r"SOURCE:\s*(https?://\S+)", text)
    return match.group(1) if match else None

# --- GESTIONARE ISTORIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Funcție pentru a converti lista de mesaje într-un string formatat pentru LLM
def get_history_text():
    history = ""
    # Luăm ultimele 6 mesaje (3 schimburi) pentru a nu aglomera prompt-ul
    last_messages = st.session_state.messages[-6:]
    for msg in last_messages:
        role = "Utilizator" if msg["role"] == "user" else "Asistent"
        history += f"{role}: {msg['content']}\n"
    return history if history else "Nicio conversație anterioară."

# Afișăm istoricul pe interfață
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT UTILIZATOR ---
if user_input := st.chat_input("Cu ce te pot ajuta?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Gândesc..."):
            # Generăm textul istoricului
            current_history = get_history_text()
            
            # Trimitem input-ul și ISTORICUL către lanț
            # Nota: LangChain classic va injecta automat contextul, dar history trebuie pasat manual
            response = rag_chain.invoke({
                "input": user_input,
                "history": current_history
            })
            
            answer = response['answer']
            st.markdown(answer)
            
            # Afișare documente și surse
            with st.expander("📄 Vezi fragmentele consultate"):
                for i, doc in enumerate(response['context']):
                    st.info(f"**Fragment {i+1}:**\n\n{doc.page_content}")
            
            sources = list(set([get_source_from_text(d.page_content) for d in response['context'] if get_source_from_text(d.page_content)]))
            if sources:
                st.markdown("**Surse oficiale:**")
                for s in sources:
                    st.markdown(f"- [{s}]({s})")

    st.session_state.messages.append({"role": "assistant", "content": answer})