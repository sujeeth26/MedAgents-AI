"""
Evaluation Setup Checker
========================

This script checks if your environment is properly configured for evaluation.
Run this BEFORE running the evaluation to ensure everything is set up correctly.

Usage:
    python check_evaluation_setup.py
"""

import os
import sys
import subprocess

def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n🔍 Checking Environment Variables...")
    
    required_vars = [
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME',
        'OPENROUTER_API_KEY',
        'OPENROUTER_MODEL'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == f'YOUR_{var}':
            missing.append(var)
            print(f"   ❌ {var}: NOT SET")
        else:
            print(f"   ✅ {var}: Set")
    
    return len(missing) == 0, missing

def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n📦 Checking Python Packages...")
    
    required_packages = [
        'nltk',
        'rouge_score',
        'sentence_transformers',
        'matplotlib',
        'seaborn',
        'pinecone',
        'langchain',
        'langchain_community',
        'langchain_pinecone',
        'pdfplumber',
        'torch',
        'numpy',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}: Installed")
        except ImportError:
            missing.append(package)
            print(f"   ❌ {package}: NOT INSTALLED")
    
    return len(missing) == 0, missing

def check_nltk_data():
    """Check if required NLTK data is downloaded."""
    print("\n📚 Checking NLTK Data...")
    
    try:
        import nltk
        required_data = ['punkt', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
                print(f"   ✅ {data}: Downloaded")
            except LookupError:
                print(f"   ⚠️  {data}: Not found (will auto-download on first run)")
        
        return True, []
    except ImportError:
        return False, ['nltk']

def check_directories():
    """Check if required directories exist."""
    print("\n📁 Checking Directories...")
    
    required_dirs = [
        './data/raw',
        './evaluation_results'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}: Exists")
        else:
            print(f"   ⚠️  {dir_path}: Will be created")
            os.makedirs(dir_path, exist_ok=True)
            print(f"      ✅ Created: {dir_path}")
    
    return True, []

def check_pinecone_connection():
    """Check if Pinecone connection works."""
    print("\n🔗 Checking Pinecone Connection...")
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key or api_key == 'YOUR_PINECONE_API_KEY':
        print("   ⚠️  Skipping (API key not set)")
        return False, ['PINECONE_API_KEY not set']
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=api_key)
        indexes = [idx.name for idx in pc.list_indexes()]
        
        index_name = os.getenv('PINECONE_INDEX_NAME', 'medagentica')
        if index_name in indexes:
            print(f"   ✅ Index '{index_name}': Found")
            
            # Check if index has data
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors > 0:
                print(f"   ✅ Vectors in index: {total_vectors}")
                return True, []
            else:
                print(f"   ⚠️  Index is empty. Run 'python demo_ingest_pinecone.py' first")
                return False, ['Index has no vectors']
        else:
            print(f"   ❌ Index '{index_name}': NOT FOUND")
            print(f"      Available indexes: {', '.join(indexes) if indexes else 'None'}")
            print(f"      Run 'python demo_ingest_pinecone.py' to create and populate")
            return False, [f'Index {index_name} not found']
            
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False, [str(e)]

def check_gpu():
    """Check GPU availability."""
    print("\n🎮 Checking GPU Availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU Available: {gpu_name}")
            print(f"   ✅ GPU Count: {gpu_count}")
            print(f"   💡 Evaluation will use GPU for faster processing")
        else:
            print(f"   ⚠️  No GPU detected")
            print(f"   💡 Evaluation will use CPU (slower but works)")
        
        return True, []
    except ImportError:
        print(f"   ❌ PyTorch not installed")
        return False, ['torch']

def print_summary(checks_results):
    """Print summary of all checks."""
    print("\n" + "="*80)
    print("📊 SETUP SUMMARY")
    print("="*80)
    
    all_passed = all(passed for passed, _ in checks_results.values())
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n🚀 Your system is ready for evaluation!")
        print("\nNext steps:")
        print("   1. Run quick test: python quick_evaluate.py")
        print("   2. Run full test: python evaluate_rag_llm.py")
        print("\n")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n❌ Issues found:")
        
        for check_name, (passed, issues) in checks_results.items():
            if not passed and issues:
                print(f"\n{check_name}:")
                for issue in issues:
                    print(f"   • {issue}")
        
        print("\n🔧 How to fix:")
        
        # Environment variables
        env_passed, env_missing = checks_results.get('Environment Variables', (True, []))
        if not env_passed and env_missing:
            print("\n1. Set environment variables:")
            for var in env_missing:
                print(f"   export {var}='your_value_here'")
        
        # Packages
        pkg_passed, pkg_missing = checks_results.get('Python Packages', (True, []))
        if not pkg_passed and pkg_missing:
            print("\n2. Install missing packages:")
            print(f"   pip install {' '.join(pkg_missing)}")
        
        # Pinecone
        pc_passed, pc_issues = checks_results.get('Pinecone Connection', (True, []))
        if not pc_passed:
            print("\n3. Set up Pinecone:")
            print("   • Ensure PINECONE_API_KEY is correct")
            print("   • Run: python demo_ingest_pinecone.py")
        
        print("\nThen run this checker again: python check_evaluation_setup.py")
        print()

def main():
    """Run all checks."""
    print("\n" + "="*80)
    print("🔬 EVALUATION SETUP CHECKER")
    print("="*80)
    print("\nThis tool checks if your environment is ready for RAG evaluation.\n")
    
    checks = {}
    
    # Run all checks
    checks['Environment Variables'] = check_environment_variables()
    checks['Python Packages'] = check_python_packages()
    checks['NLTK Data'] = check_nltk_data()
    checks['Directories'] = check_directories()
    checks['GPU'] = check_gpu()
    checks['Pinecone Connection'] = check_pinecone_connection()
    
    # Print summary
    print_summary(checks)
    
    # Exit code
    all_passed = all(passed for passed, _ in checks.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()


