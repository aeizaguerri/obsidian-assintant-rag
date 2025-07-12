import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq
from embeddings import load_embeddings_model, get_embeddings, search_similar
from vault_vectorize import vectorize_docs

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
VAULT_PATH = "../.."

class GroqChat:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.client = Groq(api_key=api_key or GROQ_API_KEY)
        self.model = model
        self.embeddings_model = load_embeddings_model()
        self.conversation_history = []
        
        # Initialize vector database with verbose output
        print("Initializing vector database...")
        self.index, self.metadata = vectorize_docs(VAULT_PATH)
        
        if self.index is not None:
            print(f"✓ Vector database ready with {len(self.metadata)} documents")
        else:
            print("⚠ No vector database found - RAG features will be limited")
        
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from the vector database using RAG"""
        if self.index is None:
            return ""
        
        # Get query embedding
        query_embedding = get_embeddings(self.embeddings_model, query)
        
        # Search for similar documents
        results = search_similar(query_embedding, k)
        
        # Format context
        context_parts = []
        for distance, metadata in results:
            context_parts.append(f"Source: {metadata['file_path']}\nContent: {metadata['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def create_system_prompt(self, context: str) -> str:
        """Create system prompt with RAG context"""
        return f"""You are an AI assistant with access to a knowledge base. Use the following context to answer questions accurately and helpfully.

Context from knowledge base:
{context}

Instructions:
- Use the provided context to answer questions when relevant
- If the context doesn't contain relevant information, say so clearly
- Cite specific sources when referencing information from the context
- Be concise and helpful in your responses
"""
    
    def chat(self, user_message: str, use_rag: bool = True, k: int = 3) -> str | None:
        """Send a chat message with optional RAG context"""
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
        self.index, self.metadata = vectorize_docs(VAULT_PATH, force_rebuild=True)
    
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