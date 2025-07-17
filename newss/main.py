import streamlit as st
import os
from dotenv import load_dotenv
import pickle
import hashlib
import requests
import time
import logging
import datetime
import re
from gtts import gTTS
import io

# LangChain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="NewsEdge AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': "https://example.com",
        'About': "# Advanced News Intelligence Platform"
    }
)

# Custom CSS for professional dark UI
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --secondary: #2980b9;
        --accent: #1abc9c;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text: #f8fafc;
        --text-secondary: #cbd5e1;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --border: #334155;
    }
    
    body {
        background-color: var(--dark-bg);
        color: var(--text);
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 1.5rem 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border-bottom: 1px solid var(--border);
    }
    
    .app-title {
        color: var(--text);
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    
    .card-header {
        color: var(--text);
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 18px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-connected {
        background-color: rgba(16, 185, 129, 0.15);
        color: var(--success);
    }
    
    .status-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: var(--danger);
    }
    
    .stButton button {
        background: linear-gradient(135deg, var(--accent), var(--primary)) !important;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s;
        border: none;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .source-chip {
        display: inline-block;
        background-color: rgba(52, 152, 219, 0.15);
        color: var(--primary);
        padding: 5px 14px;
        border-radius: 18px;
        margin: 4px 4px 4px 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .feature-badge {
        background: linear-gradient(135deg, var(--accent), var(--primary));
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0 8px 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    
    .section-title {
        color: var(--text);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border);
    }
    
    .sentiment-tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        margin: 0 4px;
    }
    
    .sentiment-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: var(--success);
    }
    
    .sentiment-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: var(--danger);
    }
    
    .sentiment-neutral {
        background-color: rgba(245, 158, 11, 0.15);
        color: var(--warning);
    }
    
    .impact-tag {
        display: inline-block;
        background-color: rgba(139, 92, 246, 0.15);
        color: #8b5cf6;
        padding: 5px 12px;
        border-radius: 14px;
        margin: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    
    .company-tag {
        display: inline-block;
        background-color: rgba(52, 152, 219, 0.15);
        color: var(--primary);
        padding: 5px 12px;
        border-radius: 14px;
        margin: 0 4px 4px 0;
        font-size: 0.85rem;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    
    .insight-card {
        background: linear-gradient(to bottom, #1e293b, #1a2535);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border-left: 4px solid var(--accent);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .timeline-event {
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(to right, #1e293b, #1a2535);
        border-radius: 10px;
        border-left: 4px solid var(--primary);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .domain-btn {
        flex: 1;
        min-width: 100px;
        text-align: center;
        padding: 10px 15px;
        border-radius: 8px;
        background-color: #1a2535;
        color: var(--text);
        cursor: pointer;
        transition: all 0.3s;
        border: 1px solid var(--border);
        font-size: 0.95rem;
        margin: 0 5px 10px 0;
        font-weight: 500;
    }
    
    .domain-btn:hover {
        background-color: #233044;
        transform: translateY(-2px);
    }
    
    .domain-btn.active {
        background: linear-gradient(135deg, var(--accent), var(--primary));
        color: white;
        border-color: var(--primary);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        padding: 1.5rem;
        border-right: 1px solid var(--border);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .header-container {
            padding: 1.2rem;
        }
        
        .app-title {
            font-size: 1.8rem;
        }
        
        .app-subtitle {
            font-size: 0.95rem;
        }
        
        .card {
            padding: 1.2rem;
        }
        
        .feature-badge {
            font-size: 0.8rem;
            padding: 5px 10px;
        }
    }
    
    /* Input field styling */
    .stTextArea textarea {
        background-color: #1a2535 !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        padding: 12px !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    
    .stChatMessage.user {
        background-color: rgba(52, 152, 219, 0.15);
    }
    
    .stChatMessage.assistant {
        background-color: rgba(30, 41, 59, 0.8);
        border: 1px solid var(--border);
    }
    
    .stTextInput input {
        background-color: #1a2535 !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    /* Domain grid */
    .domain-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    
    /* Feature badges container */
    .features-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        margin: -1rem 0 1.5rem 0;
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
        
        # Use Kluster's custom embedding model correctly
        embeddings = OpenAIEmbeddings(
            api_key=KLUSTER_API_KEY,
            base_url=KLUSTER_BASE_URL,
            model=KLUSTER_EMBEDDING_MODEL_NAME  # Use 'model' parameter for custom models
        )
    except Exception as e:
        st.error(f"Error initializing Kluster AI services: {e}")
        st.stop()

    return llm, embeddings
    
llm, embeddings = get_kluster_llm_and_embeddings()

# --- Global Variables / Paths ---
FAISS_INDEX_DIR = "faiss_news_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Initialize session state variables
if 'summaries' not in st.session_state:
    st.session_state.summaries = []
if 'domain' not in st.session_state:
    st.session_state.domain = "Finance"
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# --- Helper Functions ---
def is_valid_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        return response.status_code == 200
    except requests.RequestException:
        return False

def analyze_sentiment(text):
    # Enhanced sentiment analysis
    positive_words = ["positive", "growth", "profit", "gain", "success", "bullish", "strong", "up", "rise", "increase"]
    negative_words = ["negative", "loss", "decline", "drop", "bearish", "crisis", "weak", "down", "fall", "decrease"]
    
    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())
    
    if positive_count > negative_count:
        return "📈 Positive", positive_count / (positive_count + negative_count + 1)
    elif negative_count > positive_count:
        return "📉 Negative", negative_count / (positive_count + negative_count + 1)
    else:
        return "😐 Neutral", 0.5

def detect_companies_and_impacts(text):
    # Improved pattern-based detection
    company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    companies = list(set(re.findall(company_pattern, text)))
    
    # Filter out common non-company words
    non_companies = ["The", "This", "That", "It", "They", "We", "You", "Article", "Report", "Analysis", "Summary"]
    companies = [c for c in companies if c not in non_companies and len(c) > 3][:5]
    
    impact_keywords = ["merger", "acquisition", "profit", "drop", "ipo", "launch", 
                      "crisis", "growth", "decline", "regulation", "investment", "expansion"]
    impacts = [word for word in impact_keywords if word in text.lower()]
    
    return companies, impacts

def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# --- LangChain Workflow Functions ---
# ... (previous code remains the same)

@st.cache_resource
def create_and_save_vector_store(urls, _embeddings_model):
    urls_string = "\n".join(sorted(urls))
    index_hash = hashlib.md5(urls_string.encode()).hexdigest()
    index_dir = os.path.join(FAISS_INDEX_DIR, f"news_index_{index_hash}")
    
    # Check if we can load existing index
    if os.path.exists(index_dir) and os.path.exists(os.path.join(index_dir, "index.faiss")):
        try:
            return FAISS.load_local(index_dir, _embeddings_model, allow_dangerous_deserialization=True)
        except Exception:
            logging.warning("Failed to load cached index, rebuilding.")

    with st.spinner("🔍 Validating URLs..."):
        valid_urls = [url for url in urls if is_valid_url(url)]
    
    if not valid_urls:
        raise ValueError("No valid URLs provided or all URLs failed to connect. Please check the links.")

    loader = UnstructuredURLLoader(urls=valid_urls, continue_on_failure=True)
    try:
        with st.spinner("📥 Fetching content from URLs..."):
            data = loader.load()
        if not data:
            raise ValueError("Could not retrieve any content from the provided URLs.")
    except Exception as e:
        raise ConnectionError(f"Failed during content loading: {e}")

    with st.spinner("✂️ Splitting documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
    if not docs:
        raise ValueError("Failed to split documents. The content might be empty.")
    
    try:
        with st.spinner("🧠 Creating vector embeddings..."):
            vectorstore = FAISS.from_documents(docs, _embeddings_model)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector embeddings. Error: {e}")

    try:
        # Save using FAISS's built-in method instead of pickling
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_dir)
    except Exception as e:
        logging.error(f"Failed to save vector store: {e}")

    return vectorstore

# ... (rest of the code remains the same)
def get_qa_chain(_vectorstore, _llm_model):
    if not _vectorstore or not _llm_model:
        return None
        
    question_generator_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:"""
    question_generator_prompt = PromptTemplate.from_template(question_generator_prompt_template)

    qa_combine_prompt_template = """You are an expert news analyst. Provide accurate and detailed answers based ONLY on the context provided.
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

def get_fast_summary_chain(_llm_model):
    summary_template = """You are an expert news summarizer. Create a concise executive summary based on the provided text from news articles.
    Focus on the main themes and key developments mentioned. Keep it to 3-4 paragraphs.
    
    Text:
    {text}
    
    Executive Summary:
    """
    summary_prompt = PromptTemplate.from_template(summary_template)
    return LLMChain(llm=_llm_model, prompt=summary_prompt)

def get_deep_summary_chain(_llm_model):
    summary_template = """You are an expert news analyst. Create a comprehensive analysis based on the provided text from news articles.
    Structure your response with the following sections:
    1. Key Developments: What are the main events or changes described?
    2. Implications: What are the potential consequences or impacts?
    3. Stakeholders: Who is affected by these developments?
    4. Critical Analysis: What are the strengths and weaknesses of the arguments presented?
    5. TL;DR: A 2-sentence summary of the most important points
    
    Text:
    {text}
    
    Comprehensive Analysis:
    """
    summary_prompt = PromptTemplate.from_template(summary_template)
    return LLMChain(llm=_llm_model, prompt=summary_prompt)

# --- Streamlit UI Components ---
st.markdown("""<div class="header-container"><div class="app-title">NewsEdge AI</div><div class="app-subtitle">Multi-Domain News Intelligence Platform</div></div>""", unsafe_allow_html=True)
st.markdown("""<div class="features-container">
    <span class="feature-badge">AI-Powered Insights</span>
    <span class="feature-badge">Sentiment Analysis</span>
    <span class="feature-badge">Impact Detection</span>
    <span class="feature-badge">Voice Summaries</span>
</div>""", unsafe_allow_html=True)

# Use sidebar for navigation and status
with st.sidebar:
    st.markdown("## Navigation")
    
    # Domain Selection
    st.markdown("### 🌐 News Domain")
    domains = ["Finance", "Politics", "Technology", "Health", "Global", "Science"]
    
    # Create domain buttons in a grid
    st.markdown('<div class="domain-grid">', unsafe_allow_html=True)
    for domain in domains:
        if st.button(domain, key=f"domain_{domain}", 
                     type="primary" if st.session_state.domain == domain else "secondary"):
            st.session_state.domain = domain
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"**Selected:** `{st.session_state.domain}`")
    st.divider()
    
    # System Status
    st.markdown("### ⚙️ System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("**AI Service**")
        st.markdown('<span class="status-badge status-connected">CONNECTED</span>' if llm else '<span class="status-badge status-error">ERROR</span>', unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("**Embeddings**")
        st.markdown('<span class="status-badge status-connected">ACTIVE</span>' if embeddings else '<span class="status-badge status-error">OFFLINE</span>', unsafe_allow_html=True)
    
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        st.markdown("**Knowledge Base**")
        st.markdown('<span class="status-badge status-connected">LOADED</span>', unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.last_processed_urls_string.splitlines())} articles")
    
    st.divider()
    
    # System Controls
    st.markdown("### ⚙️ System Controls")
    if st.button("🔄 Reset System", use_container_width=True, key="clear_kb"):
        if os.path.exists(FAISS_INDEX_DIR):
            for file_name in os.listdir(FAISS_INDEX_DIR):
                if file_name.endswith(".pkl"):
                    os.remove(os.path.join(FAISS_INDEX_DIR, file_name))
        
        keys_to_clear = ['vector_store', 'urls_processed', 'last_processed_urls_string', 'messages', 'overall_summary', 'summaries']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.cache_resource.clear()
        st.success("System reset completed")
        time.sleep(1)
        st.rerun()

    st.divider()
    
    # Indexed Sources
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        st.markdown("### 📌 Source Articles")
        urls = st.session_state.last_processed_urls_string.splitlines()
        for i, url in enumerate(urls[:3]):  # Show first 3
            st.markdown(f"<div style='font-size: 0.85rem; margin-bottom: 6px;'>{i+1}. {url}</div>", unsafe_allow_html=True)
        if len(urls) > 3:
            with st.expander(f"Show all ({len(urls)})"):
                for i, url in enumerate(urls[3:]):
                    st.markdown(f"<div style='font-size: 0.85rem; margin-bottom: 6px;'>{i+4}. {url}</div>", unsafe_allow_html=True)

# Main content area
st.markdown('<div class="section-title">📥 Source Input</div>', unsafe_allow_html=True)
with st.container():
    st.markdown("""<div class="card">""", unsafe_allow_html=True)
    urls_input = st.text_area("Enter news URLs (one per line):", height=150, 
                             placeholder="https://www.example.com/article1\nhttps://www.example.com/article2", 
                             key="url_input")
    
    process_col, _ = st.columns([1, 3])
    with process_col:
        if st.button("PROCESS ARTICLES", use_container_width=True, key="process_articles"):
            if not urls_input.strip():
                st.warning("Please enter at least one valid URL")
            else:
                current_urls_list = sorted([url.strip() for url in urls_input.split('\n') if url.strip()])
                try:
                    with st.spinner("Processing articles..."):
                        vector_store_temp = create_and_save_vector_store(current_urls_list, embeddings)
                        st.session_state.vector_store = vector_store_temp
                        st.session_state.urls_processed = True
                        st.session_state.last_processed_urls_string = "\n".join(current_urls_list)
                        st.session_state.messages = []
                        st.session_state.overall_summary = ""
                    st.success("Knowledge base created successfully")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Content Analysis Section
if 'vector_store' in st.session_state and st.session_state.vector_store:
    st.markdown('<div class="section-title">📊 Content Analysis</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""<div class="card">""", unsafe_allow_html=True)
        st.markdown("""<div class="card-header">ANALYTICAL TOOLS</div>""", unsafe_allow_html=True)
        
        # Analysis Type Selection
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ FAST SUMMARY", use_container_width=True, key="fast_summary"):
                with st.spinner("Generating concise summary..."):
                    try:
                        all_docs = list(st.session_state.vector_store.docstore._dict.values())
                        if all_docs:
                            full_text = "\n\n".join([doc.page_content for doc in all_docs])
                            summary_chain = get_fast_summary_chain(llm)
                            st.session_state.overall_summary = summary_chain.run({"text": full_text})
                            st.session_state.summary_type = "Fast"
                            st.rerun()
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
            
        with col2:
            if st.button("🎯 DEEP ANALYSIS", use_container_width=True, key="deep_summary"):
                with st.spinner("Performing comprehensive analysis..."):
                    try:
                        all_docs = list(st.session_state.vector_store.docstore._dict.values())
                        if all_docs:
                            full_text = "\n\n".join([doc.page_content for doc in all_docs])
                            summary_chain = get_deep_summary_chain(llm)
                            st.session_state.overall_summary = summary_chain.run({"text": full_text})
                            st.session_state.summary_type = "Deep"
                            st.rerun()
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        
        # Display Summary and Analysis
        if st.session_state.get('overall_summary'):
            st.markdown("""<div class="insight-card">""", unsafe_allow_html=True)
            
            # Sentiment Analysis
            sentiment, polarity = analyze_sentiment(st.session_state.overall_summary)
            
            # Company and Impact Detection
            companies, impacts = detect_companies_and_impacts(st.session_state.overall_summary)
            
            # Display metadata
            st.markdown(f"**Analysis Type:** {st.session_state.get('summary_type', 'N/A')}")
            sentiment_class = "sentiment-positive" if "Positive" in sentiment else "sentiment-negative" if "Negative" in sentiment else "sentiment-neutral"
            st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
            
            if companies:
                st.markdown("**Key Entities:**")
                for company in companies[:5]:  # Limit to 5 companies
                    st.markdown(f"<div class='company-tag'>{company}</div>", unsafe_allow_html=True)
            
            if impacts:
                st.markdown("**Impact Areas:**")
                for impact in impacts:
                    st.markdown(f"<div class='impact-tag'>{impact.capitalize()}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Summary content
            st.markdown("**Summary:**")
            st.markdown(st.session_state.overall_summary)
            
            # Human-in-the-loop Controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👍 Approve", use_container_width=True):
                    st.session_state.feedback['summary'] = "Approved"
                    st.success("Feedback recorded!")
            with col2:
                if st.button("👎 Revise", use_container_width=True):
                    st.session_state.feedback['summary'] = "Needs Revision"
                    st.warning("Feedback recorded!")
            with col3:
                if st.button("🔄 Regenerate", use_container_width=True):
                    del st.session_state.overall_summary
                    st.rerun()
            
            # Additional features
            col1, col2 = st.columns(2)
            with col1:
                # Text-to-Speech
                if st.button("🔊 Listen to Summary", key="tts", use_container_width=True, 
                            help="Convert summary to audio"):
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech(st.session_state.overall_summary)
                        st.audio(audio_bytes, format='audio/mp3')
            
            with col2:
                # Save to timeline
                if st.button("💾 Save to Timeline", use_container_width=True):
                    timestamp = datetime.datetime.now()
                    summary_entry = {
                        "timestamp": timestamp,
                        "domain": st.session_state.domain,
                        "summary": st.session_state.overall_summary,
                        "sentiment": sentiment,
                        "companies": companies,
                        "type": st.session_state.get('summary_type', 'N/A')
                    }
                    st.session_state.summaries.append(summary_entry)
                    st.success(f"Summary saved to timeline at {timestamp.strftime('%H:%M:%S')}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Timeline View
    if st.session_state.summaries:
        st.markdown('<div class="section-title">⏳ Analysis Timeline</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown("""<div class="card">""", unsafe_allow_html=True)
            st.markdown("""<div class="card-header">HISTORICAL INSIGHTS</div>""", unsafe_allow_html=True)
            
            # Group by date
            summaries_by_date = {}
            for summary in st.session_state.summaries:
                date_str = summary['timestamp'].strftime('%Y-%m-%d')
                if date_str not in summaries_by_date:
                    summaries_by_date[date_str] = []
                summaries_by_date[date_str].append(summary)
            
            # Display timeline
            for date_str, daily_summaries in sorted(summaries_by_date.items(), reverse=True):
                st.subheader(date_str)
                
                for summary in daily_summaries:
                    with st.container():
                        st.markdown(f"""<div class="timeline-event">""", unsafe_allow_html=True)
                        
                        sentiment_class = "sentiment-positive" if "Positive" in summary['sentiment'] else "sentiment-negative" if "Negative" in summary['sentiment'] else "sentiment-neutral"
                        
                        st.markdown(f"**{summary['timestamp'].strftime('%H:%M')}** - {summary['type']} Analysis")
                        st.markdown(f"**Domain:** {summary['domain']}")
                        st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{summary['sentiment']}</span>", unsafe_allow_html=True)
                        
                        if summary['companies']:
                            st.markdown("**Key Entities:**")
                            for company in summary['companies'][:3]:  # Limit to 3 companies
                                st.markdown(f"<div class='company-tag'>{company}</div>", unsafe_allow_html=True)
                        
                        with st.expander("View Summary"):
                            st.markdown(summary['summary'])
                            
                        st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Research Assistant
    st.markdown('<div class="section-title">💬 Research Assistant</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""<div class="card">""", unsafe_allow_html=True)
        st.markdown("""<div class="card-header">ASK QUESTIONS</div>""", unsafe_allow_html=True)
        if "messages" not in st.session_state or not st.session_state.messages:
            st.session_state.messages = [{"role": "assistant", "content": "Analysis complete. How can I assist with the articles?", "sources": []}]

        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        st.markdown("**Sources:**")
                        for source in msg["sources"]:
                            st.markdown(f"<div class='source-chip'>{source}</div>", unsafe_allow_html=True)
            else:
                with st.chat_message("user"):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the articles..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    qa_chain = get_qa_chain(st.session_state.vector_store, llm)
                    if qa_chain:
                        response_stream = qa_chain.stream({"question": prompt, "chat_history": st.session_state.messages})
                        for chunk in response_stream:
                            if "answer" in chunk:
                                full_response += chunk["answer"]
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        
                        retriever = st.session_state.vector_store.as_retriever()
                        source_docs = retriever.get_relevant_documents(prompt)
                        sources = list(set([doc.metadata.get('source') for doc in source_docs if doc.metadata.get('source')]))
                        if sources:
                            st.markdown("**Sources:**")
                            for source in sources[:3]:  # Limit to 3 sources
                                st.markdown(f"<div class='source-chip'>{source}</div>", unsafe_allow_html=True)
                    else:
                        full_response = "Analysis engine not available"
                        message_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"Query error: {e}"
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources if sources else []})
        st.markdown("</div>", unsafe_allow_html=True)
