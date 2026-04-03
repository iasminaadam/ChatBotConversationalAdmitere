import os
import re
import warnings
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import unicodedata

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="TUIASI Admission Bot", page_icon="🎓")
st.title("🎓 TUIASI Admission Assistant 2026")
st.markdown("Întreabă-mă orice despre admitere, taxe sau documente!")

# --- LOGICĂ BACKEND ---
@st.cache_resource
def init_rag_components():
    # Silence noisy (but harmless) transformers import warnings.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings(
        "ignore",
        message=r"Accessing `__path__` from .*",
    )

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
    return vector_db, combine_docs_chain

vector_db, combine_docs_chain = init_rag_components()

FACULTY_TAGS = [
    "Automatică și Calculatoare (AC)",
    "Arhitectură (ARH)",
    "Inginerie Chimică și Protecția Mediului (ICPM)",
    "Construcții și Instalații (CI)",
    "Design Industrial și Managementul Afacerilor (DIMA)",
    "Inginerie Electrică, Energetică și Informatică Aplicată (IEEIA)",
    "Electronică, Telecomunicații și Tehnologii Informaționale (ETTI)",
    "Hidrotehnică, Geodezie și Ingineria Mediului (HGIM)",
    "Mecanică (MEC)",
    "Știința și Ingineria Materialelor (SIM)",
    "Construcții de Mașini și Management Industrial (CMMI)",
    "TUIASI",
]

CATEGORY_TAGS = ["licenta", "master", "doctorat", "candidati", "taxe", "studii", "english"]


def _strip_diacritics(s: str) -> str:
    # "Automatică" -> "Automatica"
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize(s: str) -> str:
    return _strip_diacritics(s).lower()


def _tokens(s: str) -> list[str]:
    # Whole-token matching: "uto" won't match "automatica"
    return [t for t in re.split(r"[^a-z0-9]+", _normalize(s)) if t]


def _tag_tokens(tag: str) -> set[str]:
    # Include both words and abbreviations in parentheses.
    # "Automatică și Calculatoare (AC)" -> {"automatica","si","calculatoare","ac"}
    tokens = set(_tokens(tag))
    # Romanian stopwords that would cause false tag matches (e.g. "si" -> matches everything).
    stop_tokens = {
        "si",
        "de",
        "la",
        "in",
        "cu",
        "pe",
        "din",
        "sau",
        "pentru",
        "pt",
    }
    return {t for t in tokens if t not in stop_tokens}


_TAG_TO_TOKENS: dict[str, set[str]] = {t: _tag_tokens(t) for t in (FACULTY_TAGS + CATEGORY_TAGS)}


def extract_query_tags(query: str) -> list[str]:
    """
    Returns a list of canonical tags (strings) that appear in the query
    by whole-token matching (case/diacritics insensitive).
    """
    q_tokens = set(_tokens(query))
    matched = []
    for tag, toks in _TAG_TO_TOKENS.items():
        if toks and (q_tokens & toks):
            matched.append(tag)
    return matched


def build_chroma_filter_any(matched_tags: list[str]):
    """
    ChromaDB metadata contains:
    - tags: list[str]
    - source: str

    We filter on tags list membership using $contains.
    """
    if not matched_tags:
        return None
    return {"$or": [{"tags": {"$contains": t}} for t in matched_tags]}


def build_chroma_filter_all(matched_tags: list[str]):
    """
    Require that a document's metadata.tags contains ALL detected tags.
    """
    if not matched_tags:
        return None
    return {"$and": [{"tags": {"$contains": t}} for t in matched_tags]}

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
    # History should contain only prior turns (not the just-entered question).
    current_history = get_history_text()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Gândesc..."):
            matched_tags = extract_query_tags(user_input)

            # Retrieval strategy:
            # - If the question contains recognizable tags, we FILTER to those docs (higher precision).
            # - Otherwise, fall back to normal retrieval over all docs.
            chroma_filter_all = build_chroma_filter_all(matched_tags)
            chroma_filter_any = build_chroma_filter_any(matched_tags)

            if chroma_filter_all:
                docs = vector_db.similarity_search(user_input, k=10, filter=chroma_filter_all)
                # If too strict (no docs contain all tags), relax to any-tag matching.
                if not docs and chroma_filter_any:
                    docs = vector_db.similarity_search(user_input, k=10, filter=chroma_filter_any)
            else:
                docs = vector_db.similarity_search(user_input, k=10)

            answer = combine_docs_chain.invoke(
                {
                    "input": user_input,
                    "history": current_history,
                    "context": docs,
                }
            )
            st.markdown(answer)
            
            # Afișare documente și surse
            with st.expander("📄 Vezi fragmentele consultate"):
                if matched_tags:
                    st.caption(f"Filtrare după tag-uri detectate: {', '.join(matched_tags)}")
                for i, doc in enumerate(docs):
                    st.info(f"**Fragment {i+1}:**\n\n{doc.page_content}")
            
            sources = []
            for d in docs:
                src = (d.metadata or {}).get("source")
                if src:
                    sources.append(src)
            sources = list(dict.fromkeys(sources))  # stable unique
            if sources:
                st.markdown("**Surse oficiale:**")
                for s in sources:
                    st.markdown(f"- [{s}]({s})")

    st.session_state.messages.append({"role": "assistant", "content": answer})