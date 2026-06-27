import os
import sys
import shutil
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logger import log
from src.data_loader import DataFilesLoader
from src.chunks import GenerateChunks
from src.basestore import FaissVectorStore, BM25Store
from src.hybridretriever import HybridRetiever
from src.llm import GroqLLM
from src.config import GORQ_API , CHUNK_OVERLAP ,CHUNK_SIZE , DOC_PATH

st.set_page_config(
    page_title="RAG Context Bot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #818cf8 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        
        .subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

UPLOAD_DIR = f"rootdata/{DOC_PATH}"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.markdown('<div class="main-title"><b>SparkPlug<b></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Search, retrieve and chat with uploaded documents using Hybrid BM25 & Vector Store.</div>', unsafe_allow_html=True)


api_key = GORQ_API

st.sidebar.title("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "csv", "txt", "docx", "xlsx", "json"],
    accept_multiple_files=True
)

if uploaded_files:
    saved = False
    for uploaded_file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved = True
    if saved:
        st.sidebar.success("New files uploaded successfully!")
        st.cache_resource.clear()
        st.rerun()


st.sidebar.markdown("### Active Files")
if os.path.exists(UPLOAD_DIR):
    files = os.listdir(UPLOAD_DIR)
    if files:
        for file in files:
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            col1.text(f"📄 {file}")
            if col2.button("🗑️", key=f"del_{file}"):
                try:
                    os.remove(os.path.join(UPLOAD_DIR, file))
                    if os.path.exists("vectorstore"):
                        shutil.rmtree("vectorstore")
                    st.cache_resource.clear()
                    st.sidebar.success(f"Deleted {file}")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.info("No documents uploaded.")



@st.cache_resource(show_spinner=True)
def build_pipeline():
    
    log.info("Rebuilding RAG index from data extractor...")

    loader = DataFilesLoader()
    documents = loader.dataExtractor(DOC_PATH)
    if not documents:
        raise ValueError("Could not extract any documents from rootdata/temp_upload.")

    gen_chunks = GenerateChunks(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = gen_chunks.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks created from documents.")

    vectorstore = FaissVectorStore("vectorstore", chunks=chunks)
    bm25 = BM25Store("vectorstore", chunks=chunks)

    retriever = HybridRetiever(vectorstore, bm25)
    llm = GroqLLM()
    
    return retriever, llm



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

status = st.sidebar.empty()
retriever = None
llm = None


if not api_key:
    status.error("❌ GORQ_API Key Missing")
    st.warning("⚠️ Please provide a valid Groq API Key in the sidebar control panel.")
elif not os.listdir(UPLOAD_DIR):
    status.error("❌ No Documents Uploaded")
else:
    status.info("⏳ Initializing RAG components...")
    try:
        with st.spinner("Building vector database and loading sentence transformer models..."):
            retriever, llm = build_pipeline()
            status.success("✅ System Ready")
    except Exception as e:
        status.error("❌ Pipeline Failed")
        st.error(f"Failed to load RAG classes: {e}")
        st.info("Check if your files are formatted correctly or if your API key is correct.")


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "refs" in msg and msg["refs"]:
            with st.expander("📚 Sources & References"):
                for idx, ref in enumerate(msg["refs"]):
                    st.markdown(f"**Source {idx+1}:** {ref['source']} (Page: {ref['page']})")
                    st.caption(ref['content'])


if retriever and llm:
    if query := st.chat_input("Enter your question..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving answers..."):
                try:

                    results = retriever.query(query)
                    
                    refs = []
                    for doc_dict in results:
                        doc = doc_dict.get("document")
                        if doc:
                            refs.append({
                                "source": doc.metadata.get("source_file", doc.metadata.get("source", "unknown")),
                                "page": doc.metadata.get("page", "unknown"),
                                "content": doc.page_content
                            })
                            
   
                    response = llm.generate(query=query, documents=results)

                    if hasattr(response, "content"):
                        response_text = response.content
                    else:
                        response_text = str(response)
                        
                    st.markdown(response_text)
                    
                    if refs:
                        with st.expander("📚 Sources & References"):
                            for idx, ref in enumerate(refs):
                                st.markdown(f"**Source {idx+1}:** {ref['source']} (Page: {ref['page']})")
                                st.caption(ref['content'])
                                
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "refs": refs
                    })
                except Exception as e:
                    st.error(f"Error querying response: {e}")
                    log.error(f"Query Error: {e}")
else:
    st.chat_input("Pipeline offline. Please solve status issues to start.", disabled=True)
