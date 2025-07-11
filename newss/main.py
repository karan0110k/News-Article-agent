import streamlit as st
import os
from dotenv import load_dotenv
import pickle
import hashlib
import requests
import time
import logging

# LangChain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="NewsEdge AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': "https://example.com",
        'About': "# Your edge through news and AI"
    }
)

# Custom CSS for modern UI
st.markdown("""
<style>
    :root {
        --primary: #1a5276;
        --secondary: #2980b9;
        --accent: #3498db;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #27ae60;
        --warning: #f39c12;
        --danger: #e74c3c;
    }
    
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 2rem 1.5rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top:-50px;
    }
    
    .app-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .app-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.4rem;
        font-weight: 300;
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .card-header {
        color: var(--primary);
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .card-header i {
        margin-right: 10px;
        font-size: 1.4rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-connected {
        background-color: rgba(39, 174, 96, 0.15);
        color: var(--success);
    }
    
    .status-error {
        background-color: rgba(231, 76, 60, 0.15);
        color: var(--danger);
    }
    
    .stButton button {
        background: linear-gradient(135deg, var(--accent), var(--secondary)) !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(50, 152, 220, 0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(50, 152, 220, 0.3);
        background: linear-gradient(135deg, var(--secondary), var(--primary)) !important;
    }
    
    .source-chip {
        display: inline-block;
        background-color: rgba(52, 152, 219, 0.15);
        color: var(--primary);
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px 5px 5px 0;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .source-chip:hover {
        background-color: rgba(52, 152, 219, 0.25);
        transform: translateY(-2px);
    }
    
    .assistant-msg {
        background: linear-gradient(to right, rgba(52, 152, 219, 0.1), rgba(52, 152, 219, 0.05));
        padding: 1.25rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        border-left: 3px solid var(--accent);
    }
    
    .user-msg {
        background-color: rgba(234, 236, 240, 0.5);
        padding: 1.25rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .summary-box {
        background: linear-gradient(to right, rgba(243, 156, 18, 0.08), rgba(243, 156, 18, 0.03));
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid var(--warning);
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(52, 152, 219, 0.3), transparent);
        margin: 1.8rem 0;
    }
    
    .section-title {
        color: var(--primary);
        font-size: 1.6rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(52, 152, 219, 0.2);
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .feature-badge {
        background: linear-gradient(135deg, var(--accent), var(--secondary));
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
KLUSTER_API_KEY = os.getenv("KLUSTER_API_KEY")
KLUSTER_BASE_URL = os.getenv("KLUSTER_BASE_URL")
KLUSTER_CHAT_MODEL_NAME = os.getenv("KLUSTER_CHAT_MODEL_NAME")
KLUSTER_EMBEDDING_MODEL_NAME = os.getenv("KLUSTER_EMBEDDING_MODEL_NAME")

# --- Initialize LLM and Embeddings (Cached for efficiency) ---
@st.cache_resource
def get_kluster_llm_and_embeddings():
    llm = None
    embeddings = None

    if not all([KLUSTER_API_KEY, KLUSTER_BASE_URL, KLUSTER_CHAT_MODEL_NAME, KLUSTER_EMBEDDING_MODEL_NAME]):
        st.error("One or more Kluster AI credentials are missing from your .env file. Please check.")
        st.stop()
        
    try:
        llm = ChatOpenAI(
            api_key=KLUSTER_API_KEY,
            base_url=KLUSTER_BASE_URL,
            model=KLUSTER_CHAT_MODEL_NAME,
            temperature=0.3,
            streaming=True
        )
        embeddings = OpenAIEmbeddings(
            api_key=KLUSTER_API_KEY,
            base_url=KLUSTER_BASE_URL,
            model=KLUSTER_EMBEDDING_MODEL_NAME
        )
    except Exception as e:
        st.error(f"Error initializing Kluster AI services: {e}")
        st.stop()

    return llm, embeddings

llm, embeddings = get_kluster_llm_and_embeddings()

# --- Global Variables / Paths ---
FAISS_INDEX_DIR = "faiss_news_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# --- Helper Functions ---
def is_valid_url(url):
    try:
        # UPDATED: Use a standard User-Agent to avoid simple bot blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        return response.status_code == 200
    except requests.RequestException:
        return False

# --- LangChain Workflow Functions ---
# UPDATED: This function now raises exceptions instead of returning None silently.
@st.cache_resource(hash_funcs={OpenAIEmbeddings: lambda _: None})
def create_and_save_vector_store(urls, _embeddings_model):
    urls_string = "\n".join(sorted(urls))
    index_hash = hashlib.md5(urls_string.encode()).hexdigest()
    index_file_path = os.path.join(FAISS_INDEX_DIR, f"news_index_{index_hash}.pkl")

    if os.path.exists(index_file_path):
        try:
            with open(index_file_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            logging.warning("Failed to load cached index, rebuilding.")

    with st.spinner("Validating URLs..."):
        valid_urls = [url for url in urls if is_valid_url(url)]
    
    if not valid_urls:
        raise ValueError("No valid URLs provided or all URLs failed to connect. Please check the links.")

    loader = UnstructuredURLLoader(urls=valid_urls, continue_on_failure=True)
    try:
        with st.spinner("Fetching content from URLs..."):
            data = loader.load()
        if not data:
            raise ValueError("Could not retrieve any content from the provided URLs. The sites may be blocking scrapers or the content is not accessible.")
    except Exception as e:
        raise ConnectionError(f"Failed during content loading: {e}")

    with st.spinner("Splitting documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
    if not docs:
        raise ValueError("Failed to split documents. The content might be empty or in an unsupported format.")
    
    try:
        with st.spinner("Creating vector embeddings..."):
            vectorstore = FAISS.from_documents(docs, _embeddings_model)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector embeddings. Error: {e}")

    try:
        with open(index_file_path, "wb") as f:
            pickle.dump(vectorstore, f)
    except Exception as e:
        logging.error(f"Failed to save vector store cache: {e}")

    return vectorstore

def get_qa_chain(_vectorstore, _llm_model):
    if not _vectorstore or not _llm_model:
        return None
        
    question_generator_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:"""
    question_generator_prompt = PromptTemplate.from_template(question_generator_prompt_template)

    qa_combine_prompt_template = """You are an expert financial news analyst. Provide accurate and detailed answers based ONLY on the context provided.
    If the answer cannot be found in the context, state "I don't have information on this from the provided articles." Do not use outside knowledge.
    After providing the answer, cite your sources clearly by listing the source URLs from which you gathered the information.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    qa_combine_prompt = PromptTemplate.from_template(qa_combine_prompt_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    chain = ConversationalRetrievalChain.from_llm(
        llm=_llm_model,
        retriever=_vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=question_generator_prompt,
        combine_docs_chain_kwargs={"prompt": qa_combine_prompt},
        return_source_documents=True,
    )
    return chain

def get_overall_summary_chain(_llm_model):
    summary_template = """You are an expert financial news summarizer. Create a concise executive summary based on the provided text from news articles.
    Focus on the main themes, key financial implications, market sentiment, company-specific news, and any significant economic developments mentioned.
    Structure the output with clear headings.
    
    Text:
    {text}
    
    Executive Summary:
    """
    summary_prompt = PromptTemplate.from_template(summary_template)
    return LLMChain(llm=_llm_model, prompt=summary_prompt)

# --- Streamlit UI Components ---

st.markdown("""<div class="header-container"><div class="app-title">NewsEdge AI</div><div class="app-subtitle">Your edge through news and AI</div></div>""", unsafe_allow_html=True)
st.markdown("""<div style="margin: -1rem 0 1.5rem 0; display: flex; flex-wrap: wrap;"><span class="feature-badge">AI-Powered Analysis</span><span class="feature-badge">Financial Insights</span><span class="feature-badge">Source Verification</span><span class="feature-badge">Real-time Processing</span></div>""", unsafe_allow_html=True)

main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    with st.container():
        st.markdown("""<div class="card"><div class="card-header">⚙️ System Status</div>""", unsafe_allow_html=True)
        st.markdown("**LLM Service**")
        st.markdown('<span class="status-badge status-connected">✅ Connected</span>' if llm else '<span class="status-badge status-error">⚠️ Error</span>', unsafe_allow_html=True)
        if llm: st.caption("Kluster AI")
        st.markdown("**Embeddings Service**")
        st.markdown('<span class="status-badge status-connected">✅ Ready</span>' if embeddings else '<span class="status-badge status-error">⚠️ Error</span>', unsafe_allow_html=True)
        if embeddings: st.caption("Kluster AI")
        st.markdown("**Knowledge Base**")
        if 'vector_store' in st.session_state and st.session_state.vector_store:
            st.markdown('<span class="status-badge status-connected">✅ Loaded</span>', unsafe_allow_html=True)
            st.caption(f"{len(st.session_state.last_processed_urls_string.splitlines())} articles")
        else:
            st.markdown('<span class="status-badge status-error">⚠️ Not Loaded</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Knowledge Base & Restart", use_container_width=True):
        if os.path.exists(FAISS_INDEX_DIR):
            for file_name in os.listdir(FAISS_INDEX_DIR):
                if file_name.endswith(".pkl"):
                    os.remove(os.path.join(FAISS_INDEX_DIR, file_name))
        
        keys_to_clear = ['vector_store', 'urls_processed', 'last_processed_urls_string', 'messages', 'overall_summary']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.cache_resource.clear()
        st.rerun()

    st.markdown("""<div class="card"><div class="card-header">📚 How to Use</div><ol style="padding-left: 1.2rem; margin-top: 0.5rem;"><li style="margin-bottom: 0.5rem;">Add news article URLs below</li><li style="margin-bottom: 0.5rem;">Click 'Process Articles'</li><li style="margin-bottom: 0.5rem;">Generate insights or ask questions</li></ol><p style="margin-top: 1rem; font-size: 0.9rem;"><strong>Pro Tip:</strong> Use multiple articles from different sources for comprehensive analysis.</p></div>""", unsafe_allow_html=True)

with main_col:
    st.markdown('<div class="section-title">📥 Add News Articles</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""<div class="card"><div class="card-header">🌐 Article Sources</div><p>Add URLs of financial news articles for AI analysis:</p>""", unsafe_allow_html=True)
        urls_input = st.text_area("Enter URLs (one per line):", height=150, placeholder="https://www.reuters.com/markets/article1\nhttps://www.bloomberg.com/news/article2", label_visibility="collapsed")
        
        if st.button("🚀 Process Articles", use_container_width=True):
            if not urls_input.strip():
                st.warning("Please enter at least one URL.")
            else:
                current_urls_list = sorted([url.strip() for url in urls_input.split('\n') if url.strip()])
                # UPDATED: Implemented robust try-except block to catch specific errors
                try:
                    vector_store_temp = create_and_save_vector_store(current_urls_list, embeddings)
                    st.session_state.vector_store = vector_store_temp
                    st.session_state.urls_processed = True
                    st.session_state.last_processed_urls_string = "\n".join(current_urls_list)
                    st.session_state.messages = []
                    st.session_state.overall_summary = ""
                    st.success("✅ Knowledge base created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    if 'vector_store' in st.session_state and st.session_state.vector_store:
        st.markdown('<div class="section-title">📊 Executive Summary</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown("""<div class="card"><div class="card-header">📝 Key Insights</div>""", unsafe_allow_html=True)
            if st.button("✨ Generate Executive Summary", use_container_width=True):
                with st.spinner("🔍 Analyzing articles..."):
                    try:
                        all_docs = list(st.session_state.vector_store.docstore._dict.values())
                        if all_docs:
                            full_text = "\n\n".join([doc.page_content for doc in all_docs])
                            summary_chain = get_overall_summary_chain(llm)
                            st.session_state.overall_summary = summary_chain.run({"text": full_text})
                        else:
                            st.error("❌ No content available for summarization.")
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {e}")

            if st.session_state.get('overall_summary'):
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.overall_summary)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">💬 Research Assistant</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown("""<div class="card"><div class="card-header">🤖 Ask Questions</div>""", unsafe_allow_html=True)
            if "messages" not in st.session_state or not st.session_state.messages:
                st.session_state.messages = [{"role": "assistant", "content": "I've analyzed the articles. Ask me anything about their content."}]

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        st.markdown("**Sources:**")
                        for source in msg["sources"]:
                            st.markdown(f"<div class='source-chip'>{source}</div>", unsafe_allow_html=True)

            if prompt := st.chat_input("Ask about the articles..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        qa_chain = get_qa_chain(st.session_state.vector_store, llm)
                        if qa_chain:
                            # UPDATED: Implemented true streaming
                            response_stream = qa_chain.stream({"question": prompt, "chat_history": st.session_state.messages})
                            for chunk in response_stream:
                                if "answer" in chunk:
                                    full_response += chunk["answer"]
                                    message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                            
                            # UPDATED: Retrieve sources after the answer has been streamed
                            retriever = st.session_state.vector_store.as_retriever()
                            source_docs = retriever.get_relevant_documents(prompt)
                            sources = list(set([doc.metadata.get('source') for doc in source_docs if doc.metadata.get('source')]))
                            if sources:
                                st.markdown("**Sources:**")
                                for source in sources:
                                    st.markdown(f"<div class='source-chip'>{source}</div>", unsafe_allow_html=True)
                        else:
                            full_response = "The question-answering chain is not available."
                            message_placeholder.markdown(full_response)

                    except Exception as e:
                        full_response = f"An error occurred: {e}"
                        message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
            st.markdown("</div>", unsafe_allow_html=True)