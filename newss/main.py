import streamlit as st
import os
from dotenv import load_dotenv
import pickle
import hashlib
import requests
import time

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
        background-color: black;
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

    # Initialize Kluster AI Chat Model
    if KLUSTER_API_KEY and KLUSTER_BASE_URL and KLUSTER_CHAT_MODEL_NAME:
        try:
            llm = ChatOpenAI(
                api_key=KLUSTER_API_KEY,
                base_url=KLUSTER_BASE_URL,
                model=KLUSTER_CHAT_MODEL_NAME,
                temperature=0.3,
                streaming=True
            )
        except Exception as e:
            st.error(f"Error initializing Kluster AI LLM: {e}")
            st.stop()
    else:
        st.error("Kluster AI LLM credentials missing from .env")
        st.stop()

    # Initialize Kluster AI Embeddings Model
    if KLUSTER_API_KEY and KLUSTER_BASE_URL and KLUSTER_EMBEDDING_MODEL_NAME:
        try:
            embeddings = OpenAIEmbeddings(
                api_key=KLUSTER_API_KEY,
                base_url=KLUSTER_BASE_URL,
                model=KLUSTER_EMBEDDING_MODEL_NAME
            )
        except Exception as e:
            st.error(f"Error initializing Kluster AI Embeddings: {e}")
            st.stop()
    else:
        st.error("Kluster AI Embedding credentials missing from .env")
        st.stop()

    return llm, embeddings

llm, embeddings = get_kluster_llm_and_embeddings()

# --- Global Variables / Paths ---
FAISS_INDEX_DIR = "faiss_news_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# --- Helper Functions ---
def is_valid_url(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except:
        return False

# --- LangChain Workflow Functions ---
@st.cache_resource(hash_funcs={OpenAIEmbeddings: lambda _: None})
def create_and_save_vector_store(urls, _embeddings_model):
    urls_string = "\n".join(sorted(urls))
    index_hash = hashlib.md5(urls_string.encode()).hexdigest()
    index_file_path = os.path.join(FAISS_INDEX_DIR, f"news_index_{index_hash}.pkl")

    # Attempt to load existing knowledge base
    if os.path.exists(index_file_path):
        try:
            with open(index_file_path, "rb") as f:
                vectorstore = pickle.load(f)
            return vectorstore
        except:
            pass
    
    valid_urls = [url for url in urls if is_valid_url(url)]
    invalid_urls = [url for url in urls if not is_valid_url(url)]
    
    if not valid_urls:
        return None

    loader = UnstructuredURLLoader(urls=valid_urls)
    try:
        data = loader.load()
        if not data:
            return None
    except:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(data)
    if not docs:
        return None
    
    try:
        vectorstore = FAISS.from_documents(docs, _embeddings_model)
    except:
        return None

    try:
        with open(index_file_path, "wb") as f:
            pickle.dump(vectorstore, f)
    except:
        pass

    return vectorstore

def get_qa_chain(_vectorstore, _llm_model, _chat_history):
    if _vectorstore and _llm_model:
        question_generator_prompt_template = """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone Question:"""
        
        question_generator_prompt = PromptTemplate(
            template=question_generator_prompt_template,
            input_variables=["chat_history", "question"]
        )

        qa_combine_prompt_template = """
        You are an expert financial news analyst. Provide accurate answers based on the context.
        If the answer cannot be found in the context, state "I don't have information on this from the provided articles."
        Always cite your sources clearly by listing the URLs from which you gathered the information.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        qa_combine_prompt = PromptTemplate(
            template=qa_combine_prompt_template,
            input_variables=["context", "question"]
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

        for message in _chat_history:
            if message["role"] == "user":
                memory.chat_memory.add_user_message(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(AIMessage(content=message["content"]))

        chain = ConversationalRetrievalChain.from_llm(
            llm=_llm_model,
            retriever=_vectorstore.as_retriever(),
            memory=memory,
            condense_question_prompt=question_generator_prompt,
            combine_docs_chain_kwargs={"prompt": qa_combine_prompt},
            return_source_documents=True,
            verbose=False
        )
        return chain
    return None

def get_overall_summary_chain(_llm_model):
    summary_template = """
    You are a financial news summarizer. Summarize the main themes and key takeaways from these news articles.
    Focus on financial implications, market sentiment, company news, and significant developments.
    Provide a concise executive summary.
    
    Text:
    {text}
    
    Summary:
    """
    summary_prompt = PromptTemplate(template=summary_template, input_variables=["text"])
    chain = LLMChain(llm=_llm_model, prompt=summary_prompt)
    return chain

# --- Streamlit UI Components ---

# Modern Header
st.markdown("""
<div class="header-container">
    <div class="app-title">NewsEdge AI</div>
    <div class="app-subtitle">Your edge through news and AI</div>
</div>
""", unsafe_allow_html=True)

# Feature badges
st.markdown("""
<div style="margin: -1rem 0 1.5rem 0; display: flex; flex-wrap: wrap;">
    <span class="feature-badge">AI-Powered Analysis</span>
    <span class="feature-badge">Financial Insights</span>
    <span class="feature-badge">Source Verification</span>
    <span class="feature-badge">Real-time Processing</span>
</div>
""", unsafe_allow_html=True)

# Main columns layout
main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    st.markdown("""
    <div class="card">
        <div class="card-header">⚙️ System Status</div>
    """, unsafe_allow_html=True)
    
    # LLM Status
    st.markdown("**LLM Service**")
    if llm:
        st.markdown('<span class="status-badge status-connected">✅ Connected</span>', unsafe_allow_html=True)
        st.caption("Kluster AI")
    else:
        st.markdown('<span class="status-badge status-error">⚠️ Error</span>', unsafe_allow_html=True)
    
    # Embeddings Status
    st.markdown("**Embeddings Service**")
    if embeddings:
        st.markdown('<span class="status-badge status-connected">✅ Ready</span>', unsafe_allow_html=True)
        st.caption("Kluster AI")
    else:
        st.markdown('<span class="status-badge status-error">⚠️ Error</span>', unsafe_allow_html=True)
    
    # Knowledge Base Status
    st.markdown("**Knowledge Base**")
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        st.markdown('<span class="status-badge status-connected">✅ Loaded</span>', unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.last_processed_urls_string.splitlines())} articles")
    else:
        st.markdown('<span class="status-badge status-error">⚠️ Not Loaded</span>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close card
    
    # Clear button
    if st.button("🗑️ Clear Knowledge Base & Restart", use_container_width=True, key="clear_btn"):
        if os.path.exists(FAISS_INDEX_DIR):
            for file_name in os.listdir(FAISS_INDEX_DIR):
                file_path = os.path.join(FAISS_INDEX_DIR, file_name)
                if file_path.endswith(".pkl"):
                    os.remove(file_path)
        
        keys = ['vector_store', 'urls_processed', 'last_processed_urls_string', 'messages', 'overall_summary']
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
        
        st.cache_resource.clear()
        st.rerun()
    
    # How to Use section
    st.markdown("""
    <div class="card">
        <div class="card-header">📚 How to Use</div>
        <ol style="padding-left: 1.2rem; margin-top: 0.5rem;">
            <li style="margin-bottom: 0.5rem;">Add news article URLs in the main panel</li>
            <li style="margin-bottom: 0.5rem;">Build the knowledge base</li>
            <li style="margin-bottom: 0.5rem;">Generate insights or ask questions</li>
        </ol>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: #555;">
            <strong>Pro Tip:</strong> Use multiple articles from different sources for comprehensive analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with main_col:
    # Section 1: Article Input
    st.markdown('<div class="section-title">📥 Add News Articles</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">🌐 Article Sources</div>
            <p style="margin-bottom: 1rem;">Add URLs of financial news articles for AI analysis:</p>
        """, unsafe_allow_html=True)
        
        urls_input = st.text_area(
            "Enter URLs (one per line):",
            height=200,
            placeholder="https://www.reuters.com/markets/article1\nhttps://www.bloomberg.com/news/article2\nhttps://www.ft.com/content/article3",
            label_visibility="collapsed"
        )
        
        # Process button
        if st.button("🚀 Process Articles", use_container_width=True, key="process_main"):
            if not urls_input.strip():
                st.warning("Please enter some URLs to process")
            else:
                current_urls_list = [url.strip() for url in urls_input.split('\n') if url.strip()]
                current_urls_string = "\n".join(sorted(current_urls_list))
                
                if len(current_urls_list) < 2:
                    st.warning("For best results, add at least 2 articles")
                
                with st.spinner("Building knowledge base..."):
                    progress_bar = st.progress(0)
                    vector_store_temp = create_and_save_vector_store(current_urls_list, embeddings)
                    
                    if vector_store_temp:
                        st.session_state.vector_store = vector_store_temp
                        st.session_state.urls_processed = True
                        st.session_state.last_processed_urls_string = current_urls_string
                        st.session_state.messages = []
                        st.session_state.overall_summary = ""
                        progress_bar.progress(100)
                        st.success("✅ Knowledge base created successfully!")
                    else:
                        st.error("❌ Failed to create knowledge base. Check URLs and try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card
    
    # Only show research section if articles are processed
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        # Section 2: Executive Summary
        st.markdown('<div class="section-title">📊 Executive Summary</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-header">📝 Key Insights</div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✨ Generate Executive Summary", key="summary_btn", use_container_width=True):
                    with st.spinner("🔍 Analyzing articles..."):
                        try:
                            all_docs = []
                            if hasattr(st.session_state.vector_store, 'docstore'):
                                all_docs = list(st.session_state.vector_store.docstore._dict.values())
                            else:
                                all_docs = st.session_state.vector_store.similarity_search("the", k=10000)
                            
                            if all_docs:
                                full_text = "\n\n".join([doc.page_content for doc in all_docs])
                                if len(full_text) > 8000:
                                    full_text = full_text[:8000]
                                
                                summary_chain = get_overall_summary_chain(llm)
                                summary_result = summary_chain.run(full_text)
                                st.session_state.overall_summary = summary_result
                            else:
                                st.error("❌ No content available for summarization")
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {str(e)}")
            with col2:
                if st.button("🗑️ Clear Summary", key="clear_summary", use_container_width=True):
                    if 'overall_summary' in st.session_state:
                        st.session_state.overall_summary = ""
            
            if st.session_state.get('overall_summary'):
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.overall_summary)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ Click 'Generate Executive Summary' to create an overview of all articles")
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close card
        
        # Section 3: Research Assistant
        st.markdown('<div class="section-title">💬 Research Assistant</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="card">
                <div class="card-header">🤖 Ask Questions</div>
                <p style="margin-bottom: 1rem;">Ask specific questions about the content of the processed articles:</p>
            """, unsafe_allow_html=True)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Hello! I'm your NewsEdge AI research assistant. I've analyzed the news articles you provided. Ask me anything about their content.",
                    "sources": []
                })
            
            # Display chat messages
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                sources = message.get("sources", [])
                
                if role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(f"<div class='assistant-msg'>{content}</div>", unsafe_allow_html=True)
                        if sources:
                            st.markdown("**Sources:**")
                            for source in sources:
                                st.markdown(f"<div class='source-chip'>{source}</div>", unsafe_allow_html=True)
                else:
                    with st.chat_message("user"):
                        st.markdown(f"<div class='user-msg'>{content}</div>", unsafe_allow_html=True)
            
            # Accept user input
            if prompt := st.chat_input("Ask about the articles..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)
                
                # Get response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    sources = []
                    
                    try:
                        qa_chain = get_qa_chain(
                            st.session_state.vector_store, 
                            llm, 
                            st.session_state.messages
                        )
                        
                        if qa_chain:
                            result = qa_chain({"question": prompt, "chat_history": st.session_state.messages})
                            response = result.get("answer", "I couldn't find an answer in the provided articles.")
                            source_docs = result.get("source_documents", [])
                            sources = list(set([doc.metadata.get('source') for doc in source_docs if doc.metadata.get('source')]))
                            
                            # Simulate streaming response
                            for chunk in response.split():
                                full_response += chunk + " "
                                time.sleep(0.05)
                                message_placeholder.markdown(f"<div class='assistant-msg'>{full_response}</div>", unsafe_allow_html=True)
                            
                            if sources:
                                message_placeholder.markdown(f"<div class='assistant-msg'>{response}</div>", unsafe_allow_html=True)
                                st.markdown("**Sources:**")
                                for source in sources:
                                    st.markdown(f"<div class='source-chip'>{source}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        response = f"Error: {str(e)}"
                        message_placeholder.markdown(f"<div class='assistant-msg'>{response}</div>", unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "sources": sources
                    })
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close card

# Show message if no articles processed
