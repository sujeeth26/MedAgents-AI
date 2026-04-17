"""
Quick Evaluation Script - Test Your RAG System in Minutes
==========================================================

This is a simplified version of the evaluation framework for quick testing.

Usage:
    python quick_evaluate.py

Features:
- Tests with 3 sample medical queries
- Shows real-time progress
- Displays results immediately
- Creates visualizations and HTML report
"""

import os
import sys
from evaluate_rag_llm import RAGLLMEvaluator, create_medical_test_dataset
from demo_agentic_rag import AgenticRAGSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test():
    """Run a quick evaluation with 3 test cases."""
    
    print("\n" + "="*80)
    print("⚡ QUICK RAG EVALUATION - Testing Your System")
    print("="*80 + "\n")
    
    # Get credentials (support both Groq and OpenRouter)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medagentica")
    OPENROUTER_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("OPENROUTER_API_KEY"))
    OPENROUTER_MODEL = os.getenv("GROQ_MODEL", os.getenv("OPENROUTER_MODEL", "llama-3.3-70b-versatile"))
    
    # Validate
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment")
        print("   Set it with: export PINECONE_API_KEY='your_key_here'")
        return False
    
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not found in environment")
        print("   Set it with: export OPENROUTER_API_KEY='your_key_here'")
        return False
    
    try:
        # Initialize
        print("🚀 Initializing RAG system...")
        rag_system = AgenticRAGSystem(
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX_NAME,
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL
        )
        
        print("🔬 Initializing evaluator...")
        evaluator = RAGLLMEvaluator(rag_system, output_dir="./evaluation_results")
        
        # Get test cases (use only first 3 for quick test)
        all_tests = create_medical_test_dataset()
        test_cases = all_tests[:3]
        
        print(f"\n✅ Setup complete! Testing with {len(test_cases)} queries...\n")
        
        # Run evaluation
        results = evaluator.evaluate_dataset(test_cases, batch_size=1)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ QUICK TEST COMPLETE!")
        print("="*80)
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   • Success Rate: {results.get('success_rate', 0)*100:.1f}%")
        print(f"   • Avg BLEU Score: {results.get('accuracy_metrics', {}).get('bleu_score', {}).get('mean', 0):.3f}")
        print(f"   • Avg Semantic Similarity: {results.get('accuracy_metrics', {}).get('semantic_similarity', {}).get('mean', 0):.3f}")
        print(f"   • Avg Faithfulness: {results.get('rag_metrics', {}).get('faithfulness', {}).get('mean', 0):.3f}")
        print(f"   • Avg Hallucination: {results.get('rag_metrics', {}).get('hallucination_score', {}).get('mean', 0):.3f}")
        print(f"   • Avg Latency: {results.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0):.0f}ms")
        print(f"   • Avg Throughput: {results.get('performance_metrics', {}).get('tokens_per_second', {}).get('mean', 0):.1f} tok/s")
        
        print(f"\n📂 Results saved to: ./evaluation_results/")
        print(f"📄 Open the HTML report in your browser for detailed analysis!\n")
        
        # Quality assessment
        print("="*80)
        print("📈 QUICK ASSESSMENT:")
        print("="*80 + "\n")
        
        bleu = results.get('accuracy_metrics', {}).get('bleu_score', {}).get('mean', 0)
        faithfulness = results.get('rag_metrics', {}).get('faithfulness', {}).get('mean', 0)
        hallucination = results.get('rag_metrics', {}).get('hallucination_score', {}).get('mean', 0)
        latency = results.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0)
        
        # Accuracy
        if bleu > 0.5:
            print("✅ ACCURACY: Good - Answers are well-aligned with references")
        elif bleu > 0.3:
            print("⚠️  ACCURACY: Fair - Consider improving prompt engineering")
        else:
            print("❌ ACCURACY: Needs work - Review prompts and model selection")
        
        # Faithfulness
        if faithfulness > 0.7:
            print("✅ FAITHFULNESS: Good - Answers are well-grounded in context")
        elif faithfulness > 0.5:
            print("⚠️  FAITHFULNESS: Fair - Some answers may lack context support")
        else:
            print("❌ FAITHFULNESS: Low - High risk of hallucinations")
        
        # Hallucination
        if hallucination < 0.3:
            print("✅ HALLUCINATION: Low - System is reliable")
        elif hallucination < 0.5:
            print("⚠️  HALLUCINATION: Moderate - Monitor for accuracy")
        else:
            print("❌ HALLUCINATION: High - Strengthen grounding prompts")
        
        # Latency
        if latency < 2000:
            print("✅ LATENCY: Good - Fast response times")
        elif latency < 5000:
            print("⚠️  LATENCY: Acceptable - Consider optimization")
        else:
            print("❌ LATENCY: Slow - Needs performance tuning")
        
        print("\n" + "="*80)
        print("💡 TIP: Run 'python evaluate_rag_llm.py' for comprehensive testing")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)


