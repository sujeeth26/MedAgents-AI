"""
Interactive Demo for Agentic RAG System

This script provides an interactive command-line interface to test
the agentic RAG system with your own queries.

Usage:
    python interactive_demo.py
"""

import os
import sys
from dotenv import load_dotenv
from demo_agentic_rag import AgenticRAGSystem
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to reduce noise in interactive mode
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print("  🏥 AGENTIC RAG SYSTEM - Interactive Demo")
    print("  Medical Assistant with Multi-Agent Intelligence")
    print("="*80 + "\n")


def print_help():
    """Print help message."""
    print("\n📚 Available Commands:")
    print("  • Type your medical question to get an answer")
    print("  • 'help' - Show this help message")
    print("  • 'stats' - Show session statistics")
    print("  • 'clear' - Clear conversation history")
    print("  • 'debug on/off' - Toggle debug mode")
    print("  • 'quit' or 'exit' - Exit the program")
    print()


def format_response(response):
    """Format and print the response nicely."""
    print("\n" + "─"*80)
    print("📝 RESPONSE:")
    print("─"*80)
    print(response['response'])
    
    print("\n" + "─"*80)
    print("📊 METADATA:")
    print("─"*80)
    print(f"  • Confidence Score: {response['confidence']:.1%}")
    print(f"  • Documents Retrieved: {response['retrieved_doc_count']}")
    print(f"  • Query Type: {response['query_analysis'].get('query_type', 'N/A')}")
    print(f"  • Complexity: {response['query_analysis'].get('complexity', 'N/A')}")
    print(f"  • Reflection Iterations: {response['iterations']}")
    
    if response.get('sources'):
        print("\n  📚 Sources:")
        for i, source in enumerate(response['sources'][:3], 1):
            print(f"     {i}. {source.get('title', 'Unknown')} (Score: {source.get('score', 0):.3f})")
    
    print("─"*80 + "\n")


def main():
    """Main interactive loop."""
    # Load environment
    load_dotenv()
    
    # Print banner
    print_banner()
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medagentica")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")
    
    # Validate configuration
    if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        print("❌ Error: PINECONE_API_KEY not configured")
        print("   Please set it in your .env file\n")
        return 1
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        print("❌ Error: OPENROUTER_API_KEY not configured")
        print("   Please set it in your .env file\n")
        return 1
    
    # Initialize system
    print("🔧 Initializing Agentic RAG System...")
    try:
        rag_system = AgenticRAGSystem(
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX_NAME,
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL
        )
        print("✅ System initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("   Please check your configuration and try again.\n")
        return 1
    
    # Session state
    chat_history = []
    query_count = 0
    debug_mode = False
    
    # Show help
    print_help()
    
    # Main loop
    print("Ready for your questions! 💬\n")
    
    while True:
        try:
            # Get user input
            user_input = input("🔍 You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the Agentic RAG System!")
                print(f"📊 Session Stats: {query_count} queries processed")
                print("═"*80 + "\n")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'clear':
                chat_history = []
                query_count = 0
                print("\n✅ Conversation history cleared!\n")
                continue
            
            elif user_input.lower() == 'stats':
                print("\n" + "─"*80)
                print("📊 SESSION STATISTICS")
                print("─"*80)
                print(f"  • Queries Processed: {query_count}")
                print(f"  • Conversation Turns: {len(chat_history) // 2}")
                print(f"  • Debug Mode: {'ON' if debug_mode else 'OFF'}")
                print(f"  • Model: {OPENROUTER_MODEL}")
                print(f"  • Index: {PINECONE_INDEX_NAME}")
                print("─"*80 + "\n")
                continue
            
            elif user_input.lower().startswith('debug'):
                parts = user_input.lower().split()
                if len(parts) > 1:
                    if parts[1] == 'on':
                        debug_mode = True
                        logging.getLogger().setLevel(logging.INFO)
                        print("\n✅ Debug mode enabled\n")
                    elif parts[1] == 'off':
                        debug_mode = False
                        logging.getLogger().setLevel(logging.WARNING)
                        print("\n✅ Debug mode disabled\n")
                else:
                    print(f"\n📊 Debug mode is: {'ON' if debug_mode else 'OFF'}\n")
                continue
            
            # Process query
            print("\n🤖 Processing your query...\n")
            
            try:
                response = rag_system.query(
                    user_query=user_input,
                    chat_history=chat_history if chat_history else None,
                    max_reflection_iterations=2
                )
                
                # Format and display response
                format_response(response)
                
                # Update chat history
                chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                chat_history.append({
                    'role': 'assistant',
                    'content': response['response']
                })
                
                # Keep only last 10 exchanges (20 messages)
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]
                
                query_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
                print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            print(f"📊 Session Stats: {query_count} queries processed")
            print("═"*80 + "\n")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()
            print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




