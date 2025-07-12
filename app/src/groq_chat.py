import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq
from vector_db import load_embeddings_model, get_embeddings, search_similar
from vectorizer import vectorize_docs
from retrieval import AdvancedRetriever, RetrievalConfig

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OBSIDIAN_FOLDER = os.getenv("OBSIDIAN_FOLDER")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

class GroqChat:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.client = Groq(api_key=api_key or GROQ_API_KEY)
        self.model = model
        self.embeddings_model = load_embeddings_model()
        self.conversation_history = []
        
        # Initialize vector database with verbose output
        print(f"Initializing vector database from: {OBSIDIAN_FOLDER}")
        self.index, self.metadata = vectorize_docs(OBSIDIAN_FOLDER)
        
        if self.index is not None and self.metadata:
            print(f"✓ Vector database ready with {len(self.metadata)} documents")
            
            # Initialize advanced retriever
            retrieval_config = RetrievalConfig(
                initial_k=15,
                final_k=5,
                enable_reranking=True,
                enable_query_expansion=True,
                diversity_threshold=0.7,
                boost_related_chunks=True,
                boost_header_matches=True
            )
            self.advanced_retriever = AdvancedRetriever(retrieval_config)
            
            print("✓ RAG system initialized with advanced features")
        else:
            print("⚠ No vector database found - RAG features will be limited")
            print("  Make sure your OBSIDIAN_FOLDER path is correct and contains .md files")
            self.advanced_retriever = None
        
    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant context from the vector database using advanced RAG
        
        Args:
            query: The search query
            k: Number of results to return (passed to advanced retriever config)
        """
        if self.index is None:
            return ""
        
        # Use only advanced retrieval
        if self.advanced_retriever is not None:
            return self.advanced_retriever.get_enhanced_context(query)
        else:
            return ""
    
    
    def create_system_prompt(self, context: str) -> str:
        """Create system prompt with RAG context"""
        return f"""You are an AI assistant with access to a comprehensive Obsidian knowledge base. Use the following context to answer questions accurately and helpfully.

The context below contains ranked search results with detailed metadata including source files, sections, tags, and relevance scores.

Context from knowledge base:
{context}

Instructions:
- Use the provided context to answer questions when relevant
- Pay attention to the relevance scores and rankings when choosing which information to prioritize  
- Reference specific sources, sections, and note titles when citing information
- If multiple sources contain relevant information, synthesize them coherently
- If the context doesn't contain sufficient information, clearly state what's missing
- Consider the document structure (headers, sections) when understanding context
- Be aware of linked notes and tags when providing comprehensive answers
- Provide specific, actionable information when possible
"""
    
    def chat(self, user_message: str, use_rag: bool = True, k: int = 5) -> str | None:
        """
        Send a chat message with optional RAG context
        
        Args:
            user_message: The user's message
            use_rag: Whether to use RAG context
            k: Number of context results
        """
        try:
            messages = []
            
            # Add RAG context if enabled
            if use_rag:
                context = self.get_relevant_context(user_message, k)
                if context:
                    system_prompt = self.create_system_prompt(context)
                    messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return assistant_message
            
        except Exception as e:
            return f"Error during chat: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def rebuild_vector_db(self):
        """Rebuild the vector database"""
        self.index, self.metadata = vectorize_docs(OBSIDIAN_FOLDER, force_rebuild=True)
    
    def get_vector_db_info(self) -> Dict:
        """Get information about the vector database"""
        if self.index is None:
            return {"status": "No vector database loaded", "documents": 0}
        
        return {
            "status": "Vector database loaded",
            "documents": len(self.metadata),
            "vector_dimension": self.index.d,
            "total_vectors": self.index.ntotal
        }

def create_chat_instance() -> GroqChat:
    """Create and return a GroqChat instance"""
    return GroqChat()

def print_help():
    """Print available commands"""
    print("""
Available commands:
  /help     - Show this help message
  /quit     - Exit the chat
  /clear    - Clear conversation history
  /rebuild  - Rebuild the vector database
  /info     - Show vector database information
  /rag on   - Enable RAG (default)
  /rag off  - Disable RAG for next message
  
Simply type your message to chat with the assistant.
""")

def main():
    """Main interactive chat loop"""
    print("🤖 Obsidian Assistant RAG Chat")
    print("="*50)
    
    try:
        chat = create_chat_instance()
    except Exception as e:
        print(f"❌ Failed to initialize chat: {e}")
        return
    
    print("\nType /help for commands or start chatting!")
    print("-" * 50)
    
    rag_enabled = True
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    print("👋 Goodbye!")
                    break
                    
                elif command == '/help':
                    print_help()
                    continue
                    
                elif command == '/clear':
                    chat.clear_history()
                    print("🧹 Conversation history cleared!")
                    continue
                    
                elif command == '/rebuild':
                    print("🔄 Rebuilding vector database...")
                    chat.rebuild_vector_db()
                    print("✓ Vector database rebuilt!")
                    continue
                    
                elif command == '/info':
                    info = chat.get_vector_db_info()
                    print(f"📊 Vector Database Info:")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
                    continue
                    
                elif command == '/rag on':
                    rag_enabled = True
                    print("✓ RAG enabled")
                    continue
                    
                elif command == '/rag off':
                    rag_enabled = False
                    print("⚠ RAG disabled for next messages")
                    continue
                    
                else:
                    print(f"❌ Unknown command: {user_input}")
                    print("Type /help for available commands.")
                    continue
            
            # Send chat message
            print("🤖 Assistant: ", end="", flush=True)
            response = chat.chat(user_input, use_rag=rag_enabled)
            
            if response:
                print(response)
            else:
                print("❌ Failed to get response from assistant")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Type /help for commands or try again.")

if __name__ == "__main__":
    main()