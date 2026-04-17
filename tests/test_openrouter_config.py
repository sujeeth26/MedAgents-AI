#!/usr/bin/env python3
"""
Test script to verify OpenRouter.ai configuration with DeepSeek V3.1
"""

import os
from dotenv import load_dotenv
from config import create_llm, create_embedding_model, create_vectorstore

def test_openrouter_config():
    """Test OpenRouter.ai configuration"""
    print("🔧 Testing OpenRouter.ai Configuration...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenRouter API key is set
    openrouter_api_key = os.getenv("openrouter_api_key")
    openrouter_model = os.getenv("openrouter_model", "deepseek/deepseek-chat-v3.1:free")
    
    print(f"📋 Configuration:")
    print(f"   API Key: {'✅ Set' if openrouter_api_key and openrouter_api_key != 'YOUR_OPENROUTER_API_KEY' else '❌ Not set or placeholder'}")
    print(f"   Model: {openrouter_model}")
    print(f"   Base URL: https://openrouter.ai/api/v1")
    
    if not openrouter_api_key or openrouter_api_key == "YOUR_OPENROUTER_API_KEY":
        print("\n❌ Error: OpenRouter API key not properly configured!")
        print("Please set your actual API key in the .env file")
        return False
    
    try:
        # Test LLM creation
        print("\n🤖 Testing LLM creation...")
        llm = create_llm(temperature=0.7)
        print(f"✅ LLM created successfully: {llm.model_name}")
        
        # Test a simple query
        print("\n💬 Testing LLM response...")
        response = llm.invoke("Hello! Can you tell me what model you are?")
        print(f"✅ LLM Response: {response.content[:100]}...")
        
        # Test embedding model
        print("\n🔍 Testing embedding model...")
        embedding_model = create_embedding_model()
        print(f"✅ Embedding model created successfully: {type(embedding_model).__name__}")
        
        # Test vector store (if configured)
        print("\n🗄️ Testing vector store...")
        try:
            vectorstore = create_vectorstore(embedding_model)
            print(f"✅ Vector store created successfully: {type(vectorstore).__name__}")
        except Exception as e:
            print(f"⚠️ Vector store creation failed (this is normal if not fully configured): {str(e)}")
        
        print("\n🎉 All tests passed! Your OpenRouter.ai configuration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if your OpenRouter API key is correct")
        print("2. Verify you have credits/access to DeepSeek V3.1")
        print("3. Check your internet connection")
        print("4. Ensure all dependencies are installed: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    test_openrouter_config()
