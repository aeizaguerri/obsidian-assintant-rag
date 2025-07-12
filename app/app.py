import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from groq_chat import GroqChat

# Configure Streamlit page
st.set_page_config(
    page_title="Obsidian Assistant RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_chat():
    """Initialize the chat instance with caching"""
    try:
        return GroqChat()
    except Exception as e:
        st.error(f"Failed to initialize chat: {e}")
        return None

def display_sidebar_content(chat_instance):
    """Display sidebar content with vector DB info and controls"""
    if chat_instance is None:
        st.error("Chat not initialized")
        return
    
    # Obsidian Vault Info
    st.markdown("### 📁 Obsidian Vault")
    vault_path = os.getenv("OBSIDIAN_FOLDER", "../..")
    st.markdown(f"**Path:** `{vault_path}`")
    
    # Check vault status
    if os.path.exists(vault_path):
        st.success("✅ Vault accessible")
        # Count markdown files
        md_count = sum(1 for root, dirs, files in os.walk(vault_path) 
                      for file in files if file.endswith('.md'))
        st.info(f"📄 {md_count} markdown files found")
    else:
        st.error("❌ Vault path not found")
        st.warning("Update OBSIDIAN_FOLDER in .env file")
    
    # Vector Database Info
    st.markdown("### 📊 Vector Database")
    info = chat_instance.get_vector_db_info()
    
    with st.container():
        # Status indicator
        if "loaded" in info["status"].lower():
            st.success(f"✅ {info['status']}")
        else:
            st.warning(f"⚠️ {info['status']}")
        
        # Database metrics
        if info["documents"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", info["documents"])
            with col2:
                if "total_vectors" in info:
                    st.metric("Vectors", info["total_vectors"])
            
            if "vector_dimension" in info:
                st.metric("Dimension", info["vector_dimension"])
        else:
            st.warning("No documents loaded")
    
    # Controls Section
    st.markdown("### 🔧 Controls")
    
    if st.button("🔄 Rebuild Vector DB", use_container_width=True):
        with st.spinner("Rebuilding vector database..."):
            chat_instance.rebuild_vector_db()
            st.success("Database rebuilt!")
            st.rerun()
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        chat_instance.clear_history()
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("History cleared!")
        st.rerun()
    
    # RAG Settings
    st.markdown("### ⚙️ Settings")
    use_rag = st.toggle("Enable RAG", value=True, help="Use document context for responses")
    st.session_state.use_rag = use_rag
    
    
    # Tips Section
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Enable RAG** for document-based answers using advanced retrieval
    - **Disable RAG** for general conversation  
    - **Rebuild DB** after adding documents
    - **Clear History** to start fresh
    - **Wikilinks** `[[page]]` are processed automatically
    - **Tags** `#tag` are included in search context
    - Advanced retrieval uses multi-stage processing with reranking
    - Ask about specific notes by title or tags for best results
    """)
    
    # Model Information
    st.markdown("### 🤖 Model Info")
    st.info(f"**LLM:** {chat_instance.model}")
    st.info(f"**Embeddings:** all-MiniLM-L6-v2")

def main():
    """Main Streamlit application"""
    
    # Initialize chat
    chat = initialize_chat()
    
    if chat is None:
        st.error("Cannot start the application. Please check your configuration.")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    
    # Header
    st.title("🤖 Obsidian Assistant RAG")
    st.markdown("Ask questions about your markdown documents")
    
    # Sidebar
    with st.sidebar:
        display_sidebar_content(chat)
    
    # Main chat interface using Streamlit's built-in chat
    
    # Display chat messages using st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input using st.chat_input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat.chat(
                        prompt, 
                        use_rag=st.session_state.use_rag
                    )
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "Sorry, I couldn't generate a response."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()