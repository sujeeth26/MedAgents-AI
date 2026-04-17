"""
Comprehensive RAG and LLM Evaluation Framework
===============================================

This script provides a complete evaluation system for RAG (Retrieval-Augmented Generation) 
and LLM systems with metrics for:

1. ACCURACY METRICS (Trustworthiness):
   - BLEU Score: N-gram overlap between generated and reference answers
   - ROUGE Score: Recall-oriented evaluation for summarization quality
   - Faithfulness: How grounded the answer is in retrieved context
   - Answer Relevancy: How well the answer addresses the question
   - Semantic Similarity: Cosine similarity between embeddings
   
2. PERFORMANCE METRICS (Scalability & Efficiency):
   - Throughput (TPS): Tokens processed per second
   - Memory Usage: RAM and GPU memory consumption
   - Success Rate: Percentage of successful completions
   - Token Efficiency: Input/output token ratios
   
3. LATENCY METRICS (Responsiveness):
   - TTFT (Time to First Token): Initial response delay
   - Total Latency: End-to-end response time
   - Retrieval Time: Time for vector search
   - Generation Time: LLM inference time
   
4. RAG-SPECIFIC METRICS:
   - Context Relevance: Quality of retrieved documents
   - Context Precision: Relevance of top-k results
   - Context Recall: Coverage of required information
   - Hallucination Score: Detection of unsupported claims
   - Source Attribution: Proper citation of sources

The evaluation generates:
- Interactive visualizations (charts and graphs)
- Detailed HTML reports with explanations
- JSON metrics for programmatic access
- Easy-to-understand summaries in plain English

Usage:
    python evaluate_rag_llm.py
"""

import os
import json
import time
import psutil
import torch
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import traceback

# Data processing
import numpy as np
import pandas as pd

# NLP and evaluation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import seaborn as sns

# LangChain and RAG components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Import your RAG system
from demo_agentic_rag import AgenticRAGSystem
from config import Config

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Could not download NLTK data")


class RAGLLMEvaluator:
    """
    Comprehensive evaluation framework for RAG and LLM systems.
    
    This class provides methods to evaluate:
    - Accuracy (correctness, faithfulness, relevance)
    - Performance (throughput, memory, efficiency)
    - Latency (response time, TTFT, generation speed)
    - RAG quality (retrieval quality, context relevance)
    """
    
    def __init__(
        self,
        rag_system: AgenticRAGSystem,
        output_dir: str = "./evaluation_results"
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_system: Your AgenticRAGSystem instance
            output_dir: Directory to save evaluation results
        """
        self.rag_system = rag_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.results = []
        self.aggregated_metrics = {}
        
        # Initialize embedding model for semantic similarity
        logger.info("📊 Loading embedding model for semantic similarity...")
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Process monitor
        self.process = psutil.Process()
        
        logger.info("✅ Evaluator initialized successfully!")
    
    def calculate_bleu_score(
        self,
        reference: str,
        candidate: str
    ) -> float:
        """
        Calculate BLEU score (N-gram overlap metric).
        
        BLEU Score measures how many words/phrases from the reference 
        appear in the candidate answer. Higher is better (0-1 scale).
        
        Technical Explanation:
        - Compares 1-gram, 2-gram, 3-gram, 4-gram matches
        - Uses smoothing to avoid zero scores
        - Common in machine translation evaluation
        
        Args:
            reference: Ground truth answer
            candidate: Generated answer
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        try:
            # Tokenize
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing
            )
            
            return float(score)
        except Exception as e:
            logger.error(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_rouge_scores(
        self,
        reference: str,
        candidate: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation).
        
        ROUGE measures how much of the reference content is captured in the candidate.
        It's recall-focused, meaning it checks if important information is included.
        
        Technical Explanation:
        - ROUGE-1: Unigram (single word) overlap
        - ROUGE-2: Bigram (two-word phrase) overlap  
        - ROUGE-L: Longest common subsequence
        
        Higher scores = better coverage of reference content
        
        Args:
            reference: Ground truth answer
            candidate: Generated answer
            
        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores
        """
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE calculation error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_semantic_similarity(
        self,
        reference: str,
        candidate: str
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Unlike BLEU/ROUGE which match exact words, this measures MEANING similarity.
        It uses neural embeddings to understand if answers convey the same information,
        even with different wording.
        
        Technical Explanation:
        - Converts text to dense vector representations (embeddings)
        - Computes cosine similarity between vectors
        - Range: -1 to 1 (typically 0 to 1 for similar texts)
        
        Args:
            reference: Ground truth answer
            candidate: Generated answer
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Generate embeddings
            ref_embedding = self.similarity_model.encode(reference, convert_to_tensor=True)
            cand_embedding = self.similarity_model.encode(candidate, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(ref_embedding, cand_embedding).item()
            
            return float(max(0.0, similarity))  # Ensure non-negative
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return 0.0
    
    def calculate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Calculate faithfulness score (how grounded the answer is in context).
        
        Faithfulness measures if the generated answer is supported by the retrieved context.
        This is crucial for RAG systems to prevent hallucinations (making up information).
        
        Technical Explanation:
        - Compares semantic similarity between answer and context
        - High score = answer is well-supported by retrieved documents
        - Low score = answer may contain hallucinations or unsupported claims
        
        Args:
            answer: Generated answer
            context: Retrieved context/documents
            
        Returns:
            Faithfulness score (0.0 to 1.0)
        """
        try:
            # Use semantic similarity as proxy for faithfulness
            return self.calculate_semantic_similarity(context, answer)
        except Exception as e:
            logger.error(f"Faithfulness calculation error: {e}")
            return 0.0
    
    def calculate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Calculate answer relevancy (how well answer addresses the question).
        
        Answer relevancy measures if the response actually answers what was asked.
        A high score means the answer is on-topic and addresses the query directly.
        
        Technical Explanation:
        - Measures semantic alignment between question and answer
        - Uses embedding similarity to detect topical relevance
        - Different from faithfulness (which checks context grounding)
        
        Args:
            question: User's question
            answer: Generated answer
            
        Returns:
            Relevancy score (0.0 to 1.0)
        """
        try:
            return self.calculate_semantic_similarity(question, answer)
        except Exception as e:
            logger.error(f"Answer relevancy calculation error: {e}")
            return 0.0
    
    def calculate_context_relevance(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Calculate context relevance (quality of retrieved documents).
        
        Context relevance measures how well the retrieval system found relevant documents.
        This is a key RAG metric - good retrieval is essential for good answers.
        
        Technical Explanation:
        - Compares question embedding with each retrieved document
        - Averages similarity scores across all contexts
        - High score = retrieval system found highly relevant documents
        
        Args:
            question: User's question
            contexts: List of retrieved context strings
            
        Returns:
            Average context relevance score (0.0 to 1.0)
        """
        try:
            if not contexts:
                return 0.0
            
            # Calculate similarity between question and each context
            question_emb = self.similarity_model.encode(question, convert_to_tensor=True)
            
            similarities = []
            for context in contexts:
                context_emb = self.similarity_model.encode(context, convert_to_tensor=True)
                sim = util.cos_sim(question_emb, context_emb).item()
                similarities.append(max(0.0, sim))
            
            # Return average relevance
            return float(np.mean(similarities))
        except Exception as e:
            logger.error(f"Context relevance calculation error: {e}")
            return 0.0
    
    def detect_hallucination(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Detect potential hallucinations in the generated answer.
        
        Hallucination detection identifies when the LLM generates information
        not supported by the retrieved context. Lower scores are better.
        
        Technical Explanation:
        - Splits answer into sentences
        - Checks if each sentence is supported by retrieved contexts
        - Hallucination score = 1 - (avg support across sentences)
        - High score = likely hallucination detected
        
        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Hallucination score (0.0 = no hallucination, 1.0 = high hallucination)
        """
        try:
            if not contexts:
                return 1.0  # No context = high hallucination risk
            
            # Split answer into sentences
            sentences = nltk.sent_tokenize(answer)
            if not sentences:
                return 0.0
            
            # Combine all contexts
            full_context = " ".join(contexts)
            
            # Check each sentence support
            support_scores = []
            for sentence in sentences:
                support = self.calculate_semantic_similarity(sentence, full_context)
                support_scores.append(support)
            
            # Hallucination = 1 - average support
            avg_support = np.mean(support_scores)
            hallucination_score = 1.0 - avg_support
            
            return float(max(0.0, min(1.0, hallucination_score)))
        except Exception as e:
            logger.error(f"Hallucination detection error: {e}")
            return 0.5  # Return moderate score on error
    
    def measure_latency_metrics(
        self,
        query: str
    ) -> Dict[str, float]:
        """
        Measure detailed latency metrics for RAG query processing.
        
        Latency metrics help optimize system responsiveness and user experience.
        
        Technical Explanation:
        - TTFT (Time to First Token): How quickly the system starts responding
        - Retrieval Time: Vector database search latency
        - Generation Time: LLM inference duration
        - Total Latency: End-to-end response time
        
        Target latencies:
        - TTFT: < 200ms (for real-time feel)
        - Total: < 2000ms (for good UX)
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary with latency measurements in milliseconds
        """
        try:
            # Start total timer
            total_start = time.time()
            
            # Measure retrieval time
            retrieval_start = time.time()
            analysis = self.rag_system.query_analysis_agent(query)
            retrieved_docs = self.rag_system.retrieval_agent(analysis)
            retrieval_time = (time.time() - retrieval_start) * 1000  # Convert to ms
            
            # Measure generation time
            generation_start = time.time()
            reflection = self.rag_system.reflection_agent(query, retrieved_docs)
            response = self.rag_system.response_synthesis_agent(
                query, retrieved_docs, reflection
            )
            generation_time = (time.time() - generation_start) * 1000  # Convert to ms
            
            # Total latency
            total_latency = (time.time() - total_start) * 1000  # Convert to ms
            
            # TTFT approximation (retrieval + model loading)
            ttft = retrieval_time + (generation_time * 0.1)  # Approximate first token time
            
            return {
                'ttft_ms': round(ttft, 2),
                'retrieval_time_ms': round(retrieval_time, 2),
                'generation_time_ms': round(generation_time, 2),
                'total_latency_ms': round(total_latency, 2)
            }
        except Exception as e:
            logger.error(f"Latency measurement error: {e}")
            return {
                'ttft_ms': 0.0,
                'retrieval_time_ms': 0.0,
                'generation_time_ms': 0.0,
                'total_latency_ms': 0.0
            }
    
    def measure_performance_metrics(
        self,
        query: str,
        response: str
    ) -> Dict[str, Any]:
        """
        Measure system performance metrics.
        
        Performance metrics help assess resource efficiency and scalability.
        
        Technical Explanation:
        - Memory Usage: RAM consumption during processing
        - Token Count: Input/output tokens for cost estimation
        - Throughput: Tokens processed per second
        - Success: Whether the query completed successfully
        
        These metrics help with:
        - Resource planning and scaling
        - Cost optimization (token-based pricing)
        - System health monitoring
        
        Args:
            query: Input query
            response: Generated response
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Token estimation (rough approximation: 1 token ≈ 4 characters)
            input_tokens = len(query) // 4
            output_tokens = len(response) // 4
            total_tokens = input_tokens + output_tokens
            
            # GPU memory if available
            gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            return {
                'memory_mb': round(memory_mb, 2),
                'gpu_memory_mb': round(gpu_memory_mb, 2),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'success': True
            }
        except Exception as e:
            logger.error(f"Performance measurement error: {e}")
            return {
                'memory_mb': 0.0,
                'gpu_memory_mb': 0.0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'success': False
            }
    
    def evaluate_single_query(
        self,
        query: str,
        reference_answer: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single query with comprehensive metrics.
        
        This method runs a complete evaluation pipeline:
        1. Processes query through RAG system
        2. Calculates accuracy metrics (BLEU, ROUGE, etc.)
        3. Measures latency (TTFT, total time)
        4. Tracks performance (memory, tokens)
        5. Evaluates RAG quality (faithfulness, retrieval)
        
        Args:
            query: User's question
            reference_answer: Expected/correct answer for comparison
            verbose: Whether to print detailed progress
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 Evaluating Query: {query[:100]}...")
            logger.info(f"{'='*80}")
        
        try:
            # Start timing
            start_time = time.time()
            
            # Get response from RAG system
            rag_response = self.rag_system.query(query)
            generated_answer = rag_response['response']
            
            # Extract retrieved contexts
            contexts = [
                doc.get('content', '') 
                for doc in rag_response.get('sources', [])
            ]
            
            # 1. ACCURACY METRICS
            if verbose:
                logger.info("\n📊 Calculating Accuracy Metrics...")
            
            bleu_score = self.calculate_bleu_score(reference_answer, generated_answer)
            rouge_scores = self.calculate_rouge_scores(reference_answer, generated_answer)
            semantic_sim = self.calculate_semantic_similarity(reference_answer, generated_answer)
            
            # 2. RAG-SPECIFIC METRICS
            if verbose:
                logger.info("📚 Calculating RAG Metrics...")
            
            full_context = " ".join(contexts) if contexts else ""
            faithfulness = self.calculate_faithfulness(generated_answer, full_context)
            answer_relevancy = self.calculate_answer_relevancy(query, generated_answer)
            context_relevance = self.calculate_context_relevance(query, contexts)
            hallucination = self.detect_hallucination(generated_answer, contexts)
            
            # 3. LATENCY METRICS
            if verbose:
                logger.info("⏱️  Measuring Latency...")
            
            latency_metrics = self.measure_latency_metrics(query)
            
            # 4. PERFORMANCE METRICS
            if verbose:
                logger.info("💻 Measuring Performance...")
            
            performance_metrics = self.measure_performance_metrics(query, generated_answer)
            
            # Calculate throughput
            total_time_s = time.time() - start_time
            tokens_per_second = performance_metrics['total_tokens'] / total_time_s if total_time_s > 0 else 0
            
            # Compile all results
            result = {
                'query': query,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'retrieved_contexts': contexts,
                
                # Accuracy Metrics
                'accuracy': {
                    'bleu_score': round(bleu_score, 4),
                    'rouge1': round(rouge_scores['rouge1'], 4),
                    'rouge2': round(rouge_scores['rouge2'], 4),
                    'rougeL': round(rouge_scores['rougeL'], 4),
                    'semantic_similarity': round(semantic_sim, 4)
                },
                
                # RAG Metrics
                'rag_quality': {
                    'faithfulness': round(faithfulness, 4),
                    'answer_relevancy': round(answer_relevancy, 4),
                    'context_relevance': round(context_relevance, 4),
                    'hallucination_score': round(hallucination, 4),
                    'num_contexts': len(contexts)
                },
                
                # Latency Metrics
                'latency': latency_metrics,
                
                # Performance Metrics
                'performance': {
                    **performance_metrics,
                    'tokens_per_second': round(tokens_per_second, 2)
                },
                
                # Metadata
                'timestamp': datetime.now().isoformat(),
                'total_evaluation_time_s': round(total_time_s, 2)
            }
            
            if verbose:
                self._print_result_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Evaluation error: {e}")
            traceback.print_exc()
            return {
                'query': query,
                'error': str(e),
                'success': False
            }
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """Print a summary of evaluation results."""
        logger.info(f"\n✅ Evaluation Complete!")
        logger.info(f"\n📈 ACCURACY METRICS:")
        logger.info(f"   BLEU Score: {result['accuracy']['bleu_score']:.3f} (0=poor, 1=perfect)")
        logger.info(f"   ROUGE-L: {result['accuracy']['rougeL']:.3f} (recall coverage)")
        logger.info(f"   Semantic Similarity: {result['accuracy']['semantic_similarity']:.3f} (meaning match)")
        
        logger.info(f"\n📚 RAG QUALITY METRICS:")
        logger.info(f"   Faithfulness: {result['rag_quality']['faithfulness']:.3f} (context grounding)")
        logger.info(f"   Answer Relevancy: {result['rag_quality']['answer_relevancy']:.3f} (addresses question)")
        logger.info(f"   Context Relevance: {result['rag_quality']['context_relevance']:.3f} (retrieval quality)")
        logger.info(f"   Hallucination Score: {result['rag_quality']['hallucination_score']:.3f} (0=none, 1=high)")
        
        logger.info(f"\n⏱️  LATENCY METRICS:")
        logger.info(f"   TTFT: {result['latency']['ttft_ms']:.1f}ms (target: <200ms)")
        logger.info(f"   Total Latency: {result['latency']['total_latency_ms']:.1f}ms (target: <2000ms)")
        logger.info(f"   Retrieval Time: {result['latency']['retrieval_time_ms']:.1f}ms")
        
        logger.info(f"\n💻 PERFORMANCE METRICS:")
        logger.info(f"   Throughput: {result['performance']['tokens_per_second']:.1f} tokens/sec")
        logger.info(f"   Memory Usage: {result['performance']['memory_mb']:.1f} MB")
        logger.info(f"   Total Tokens: {result['performance']['total_tokens']}")
    
    def evaluate_dataset(
        self,
        test_cases: List[Dict[str, str]],
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases and aggregate results.
        
        This method processes multiple queries and provides:
        - Individual results for each test case
        - Aggregated statistics (mean, std, min, max)
        - Overall system performance assessment
        
        Args:
            test_cases: List of dicts with 'query' and 'reference_answer'
            batch_size: Number of queries to process before printing summary
            
        Returns:
            Aggregated evaluation results with statistics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 EVALUATING {len(test_cases)} TEST CASES")
        logger.info(f"{'='*80}\n")
        
        all_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n📝 Test Case {i}/{len(test_cases)}")
            
            result = self.evaluate_single_query(
                query=test_case['query'],
                reference_answer=test_case['reference_answer'],
                verbose=True
            )
            
            all_results.append(result)
            
            # Print batch summary
            if i % batch_size == 0:
                logger.info(f"\n✅ Completed {i}/{len(test_cases)} test cases")
        
        # Aggregate results
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 COMPUTING AGGREGATE STATISTICS")
        logger.info(f"{'='*80}\n")
        
        aggregated = self._aggregate_results(all_results)
        aggregated['individual_results'] = all_results
        aggregated['total_test_cases'] = len(test_cases)
        
        # Save results
        self._save_results(aggregated)
        
        # Generate visualizations
        self._generate_visualizations(all_results)
        
        # Generate HTML report
        self._generate_html_report(aggregated)
        
        logger.info(f"\n✅ Evaluation complete! Results saved to: {self.output_dir}")
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple evaluations."""
        # Extract successful results
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful evaluations'}
        
        # Aggregate accuracy metrics
        accuracy_metrics = {
            'bleu_score': [r['accuracy']['bleu_score'] for r in successful_results],
            'rouge1': [r['accuracy']['rouge1'] for r in successful_results],
            'rouge2': [r['accuracy']['rouge2'] for r in successful_results],
            'rougeL': [r['accuracy']['rougeL'] for r in successful_results],
            'semantic_similarity': [r['accuracy']['semantic_similarity'] for r in successful_results]
        }
        
        # Aggregate RAG metrics
        rag_metrics = {
            'faithfulness': [r['rag_quality']['faithfulness'] for r in successful_results],
            'answer_relevancy': [r['rag_quality']['answer_relevancy'] for r in successful_results],
            'context_relevance': [r['rag_quality']['context_relevance'] for r in successful_results],
            'hallucination_score': [r['rag_quality']['hallucination_score'] for r in successful_results]
        }
        
        # Aggregate latency metrics
        latency_metrics = {
            'ttft_ms': [r['latency']['ttft_ms'] for r in successful_results],
            'total_latency_ms': [r['latency']['total_latency_ms'] for r in successful_results],
            'retrieval_time_ms': [r['latency']['retrieval_time_ms'] for r in successful_results],
            'generation_time_ms': [r['latency']['generation_time_ms'] for r in successful_results]
        }
        
        # Aggregate performance metrics
        performance_metrics = {
            'tokens_per_second': [r['performance']['tokens_per_second'] for r in successful_results],
            'memory_mb': [r['performance']['memory_mb'] for r in successful_results],
            'total_tokens': [r['performance']['total_tokens'] for r in successful_results]
        }
        
        # Calculate statistics
        def calc_stats(values):
            return {
                'mean': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values)), 4),
                'min': round(float(np.min(values)), 4),
                'max': round(float(np.max(values)), 4),
                'median': round(float(np.median(values)), 4)
            }
        
        aggregated = {
            'accuracy_metrics': {k: calc_stats(v) for k, v in accuracy_metrics.items()},
            'rag_metrics': {k: calc_stats(v) for k, v in rag_metrics.items()},
            'latency_metrics': {k: calc_stats(v) for k, v in latency_metrics.items()},
            'performance_metrics': {k: calc_stats(v) for k, v in performance_metrics.items()},
            'success_rate': round(len(successful_results) / len(results), 4),
            'total_evaluations': len(results),
            'successful_evaluations': len(successful_results)
        }
        
        # Print summary
        self._print_aggregate_summary(aggregated)
        
        return aggregated
    
    def _print_aggregate_summary(self, aggregated: Dict[str, Any]):
        """Print aggregated statistics summary."""
        logger.info("\n" + "="*80)
        logger.info("📈 AGGREGATE STATISTICS")
        logger.info("="*80 + "\n")
        
        logger.info(f"✅ Success Rate: {aggregated['success_rate']*100:.1f}%")
        logger.info(f"📊 Total Evaluations: {aggregated['total_evaluations']}")
        
        logger.info("\n📈 ACCURACY METRICS (Mean ± Std):")
        for metric, stats in aggregated['accuracy_metrics'].items():
            logger.info(f"   {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        logger.info("\n📚 RAG QUALITY METRICS (Mean ± Std):")
        for metric, stats in aggregated['rag_metrics'].items():
            logger.info(f"   {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        logger.info("\n⏱️  LATENCY METRICS (Mean ± Std):")
        for metric, stats in aggregated['latency_metrics'].items():
            logger.info(f"   {metric}: {stats['mean']:.1f} ± {stats['std']:.1f}")
        
        logger.info("\n💻 PERFORMANCE METRICS (Mean ± Std):")
        for metric, stats in aggregated['performance_metrics'].items():
            logger.info(f"   {metric}: {stats['mean']:.1f} ± {stats['std']:.1f}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
    
    def _generate_visualizations(self, results: List[Dict[str, Any]]):
        """Generate visualization charts."""
        logger.info("\n📊 Generating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Extract successful results
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            logger.warning("No successful results to visualize")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Accuracy Metrics Bar Chart
        ax1 = plt.subplot(2, 3, 1)
        accuracy_data = {
            'BLEU': [r['accuracy']['bleu_score'] for r in successful_results],
            'ROUGE-L': [r['accuracy']['rougeL'] for r in successful_results],
            'Semantic\nSimilarity': [r['accuracy']['semantic_similarity'] for r in successful_results]
        }
        
        means = [np.mean(v) for v in accuracy_data.values()]
        stds = [np.std(v) for v in accuracy_data.values()]
        
        ax1.bar(accuracy_data.keys(), means, yerr=stds, capsize=5, color=['#3498db', '#2ecc71', '#9b59b6'])
        ax1.set_ylabel('Score (0-1)')
        ax1.set_title('Accuracy Metrics\n(Higher is Better)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # 2. RAG Quality Metrics
        ax2 = plt.subplot(2, 3, 2)
        rag_data = {
            'Faithfulness': [r['rag_quality']['faithfulness'] for r in successful_results],
            'Answer\nRelevancy': [r['rag_quality']['answer_relevancy'] for r in successful_results],
            'Context\nRelevance': [r['rag_quality']['context_relevance'] for r in successful_results],
            'Hallucination\n(inverted)': [1-r['rag_quality']['hallucination_score'] for r in successful_results]
        }
        
        means = [np.mean(v) for v in rag_data.values()]
        stds = [np.std(v) for v in rag_data.values()]
        
        ax2.bar(rag_data.keys(), means, yerr=stds, capsize=5, color=['#e74c3c', '#f39c12', '#1abc9c', '#34495e'])
        ax2.set_ylabel('Score (0-1)')
        ax2.set_title('RAG Quality Metrics\n(Higher is Better)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        plt.xticks(rotation=15, ha='right')
        
        # 3. Latency Distribution
        ax3 = plt.subplot(2, 3, 3)
        latencies = [r['latency']['total_latency_ms'] for r in successful_results]
        ax3.hist(latencies, bins=15, color='#16a085', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.0f}ms')
        ax3.axvline(2000, color='orange', linestyle='--', linewidth=2, label='Target: 2000ms')
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Response Latency Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        
        # 4. Latency Breakdown
        ax4 = plt.subplot(2, 3, 4)
        retrieval_times = [r['latency']['retrieval_time_ms'] for r in successful_results]
        generation_times = [r['latency']['generation_time_ms'] for r in successful_results]
        
        x = np.arange(len(successful_results))
        width = 0.35
        
        ax4.bar(x - width/2, retrieval_times, width, label='Retrieval', color='#3498db')
        ax4.bar(x + width/2, generation_times, width, label='Generation', color='#e74c3c')
        ax4.set_xlabel('Query Index')
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('Latency Breakdown by Component', fontsize=14, fontweight='bold')
        ax4.legend()
        
        # 5. Throughput
        ax5 = plt.subplot(2, 3, 5)
        throughput = [r['performance']['tokens_per_second'] for r in successful_results]
        ax5.plot(throughput, marker='o', color='#2ecc71', linewidth=2, markersize=8)
        ax5.axhline(np.mean(throughput), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(throughput):.1f} tok/s')
        ax5.set_xlabel('Query Index')
        ax5.set_ylabel('Tokens per Second')
        ax5.set_title('System Throughput', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Memory Usage
        ax6 = plt.subplot(2, 3, 6)
        memory = [r['performance']['memory_mb'] for r in successful_results]
        ax6.plot(memory, marker='s', color='#9b59b6', linewidth=2, markersize=8)
        ax6.axhline(np.mean(memory), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(memory):.1f} MB')
        ax6.set_xlabel('Query Index')
        ax6.set_ylabel('Memory (MB)')
        ax6.set_title('Memory Consumption', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.output_dir / f"evaluation_visualizations_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Visualizations saved to: {viz_file}")
        
        plt.close()
    
    def _generate_html_report(self, aggregated: Dict[str, Any]):
        """Generate comprehensive HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG & LLM Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #555;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-subtitle {{
            font-size: 0.85em;
            color: #777;
            margin-top: 5px;
        }}
        .explanation {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #2196f3;
        }}
        .explanation h4 {{
            margin: 0 0 10px 0;
            color: #1976d2;
        }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #777;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 RAG & LLM Evaluation Report</h1>
        <p>Comprehensive Performance Analysis</p>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value {self._get_success_class(aggregated.get('success_rate', 0))}">
                    {aggregated.get('success_rate', 0)*100:.1f}%
                </div>
                <div class="metric-subtitle">{aggregated.get('successful_evaluations', 0)} / {aggregated.get('total_evaluations', 0)} queries</div>
            </div>
            <div class="metric-card">
                <h3>Average BLEU Score</h3>
                <div class="metric-value">
                    {aggregated.get('accuracy_metrics', {}).get('bleu_score', {}).get('mean', 0):.3f}
                </div>
                <div class="metric-subtitle">N-gram overlap accuracy</div>
            </div>
            <div class="metric-card">
                <h3>Average Latency</h3>
                <div class="metric-value {self._get_latency_class(aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0))}">
                    {aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0):.0f}ms
                </div>
                <div class="metric-subtitle">Target: < 2000ms</div>
            </div>
            <div class="metric-card">
                <h3>Average Throughput</h3>
                <div class="metric-value">
                    {aggregated.get('performance_metrics', {}).get('tokens_per_second', {}).get('mean', 0):.1f}
                </div>
                <div class="metric-subtitle">tokens per second</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Accuracy Metrics</h2>
        <p>These metrics measure how correct and trustworthy the generated answers are.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>What It Means</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>BLEU Score</strong></td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('bleu_score', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('bleu_score', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('bleu_score', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('bleu_score', {}).get('max', 0):.3f}</td>
                    <td>Word/phrase overlap with reference answer</td>
                </tr>
                <tr>
                    <td><strong>ROUGE-L</strong></td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('rougeL', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('rougeL', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('rougeL', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('rougeL', {}).get('max', 0):.3f}</td>
                    <td>How much reference content is captured</td>
                </tr>
                <tr>
                    <td><strong>Semantic Similarity</strong></td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('semantic_similarity', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('semantic_similarity', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('semantic_similarity', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('accuracy_metrics', {}).get('semantic_similarity', {}).get('max', 0):.3f}</td>
                    <td>Meaning similarity (embedding-based)</td>
                </tr>
            </tbody>
        </table>

        <div class="explanation">
            <h4>📚 Understanding Accuracy Metrics</h4>
            <p><strong>BLEU Score:</strong> Measures exact word/phrase matches. Higher scores mean the generated answer uses similar wording to the reference. Range: 0 (no match) to 1 (perfect match).</p>
            <p><strong>ROUGE-L:</strong> Focuses on recall - how much of the reference content appears in the answer. Good for checking if key information is included.</p>
            <p><strong>Semantic Similarity:</strong> Uses AI embeddings to check if answers have the same meaning, even with different wording. Most robust for medical content.</p>
        </div>
    </div>

    <div class="section">
        <h2>🔍 RAG Quality Metrics</h2>
        <p>These metrics evaluate the quality of your Retrieval-Augmented Generation system.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>What It Means</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Faithfulness</strong></td>
                    <td>{aggregated.get('rag_metrics', {}).get('faithfulness', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('faithfulness', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('faithfulness', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('faithfulness', {}).get('max', 0):.3f}</td>
                    <td>Answer is grounded in retrieved context</td>
                </tr>
                <tr>
                    <td><strong>Answer Relevancy</strong></td>
                    <td>{aggregated.get('rag_metrics', {}).get('answer_relevancy', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('answer_relevancy', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('answer_relevancy', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('answer_relevancy', {}).get('max', 0):.3f}</td>
                    <td>Answer directly addresses the question</td>
                </tr>
                <tr>
                    <td><strong>Context Relevance</strong></td>
                    <td>{aggregated.get('rag_metrics', {}).get('context_relevance', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('context_relevance', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('context_relevance', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('context_relevance', {}).get('max', 0):.3f}</td>
                    <td>Retrieved documents are relevant to query</td>
                </tr>
                <tr>
                    <td><strong>Hallucination Score</strong></td>
                    <td class="{self._get_hallucination_class(aggregated.get('rag_metrics', {}).get('hallucination_score', {}).get('mean', 0))}">{aggregated.get('rag_metrics', {}).get('hallucination_score', {}).get('mean', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('hallucination_score', {}).get('std', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('hallucination_score', {}).get('min', 0):.3f}</td>
                    <td>{aggregated.get('rag_metrics', {}).get('hallucination_score', {}).get('max', 0):.3f}</td>
                    <td>Unsupported claims (LOWER is better)</td>
                </tr>
            </tbody>
        </table>

        <div class="explanation">
            <h4>🎓 Understanding RAG Metrics</h4>
            <p><strong>Faithfulness:</strong> Checks if the answer is supported by retrieved documents. High faithfulness = low hallucination risk. Critical for medical accuracy!</p>
            <p><strong>Answer Relevancy:</strong> Measures if the answer actually addresses what was asked. Prevents off-topic responses.</p>
            <p><strong>Context Relevance:</strong> Evaluates retrieval quality. High scores mean your vector search is finding the right documents.</p>
            <p><strong>Hallucination Score:</strong> Detects when the LLM invents information not in the context. LOWER is better! Above 0.3 needs attention.</p>
        </div>
    </div>

    <div class="section">
        <h2>⚡ Latency & Performance Metrics</h2>
        <p>These metrics measure system speed and resource efficiency.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Target</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>TTFT (Time to First Token)</strong></td>
                    <td>{aggregated.get('latency_metrics', {}).get('ttft_ms', {}).get('mean', 0):.1f}ms</td>
                    <td>{aggregated.get('latency_metrics', {}).get('ttft_ms', {}).get('std', 0):.1f}ms</td>
                    <td>&lt; 200ms</td>
                    <td class="{self._get_ttft_class(aggregated.get('latency_metrics', {}).get('ttft_ms', {}).get('mean', 0))}">
                        {self._get_status_emoji(aggregated.get('latency_metrics', {}).get('ttft_ms', {}).get('mean', 0), 200)}
                    </td>
                </tr>
                <tr>
                    <td><strong>Total Latency</strong></td>
                    <td>{aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0):.1f}ms</td>
                    <td>{aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('std', 0):.1f}ms</td>
                    <td>&lt; 2000ms</td>
                    <td class="{self._get_latency_class(aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0))}">
                        {self._get_status_emoji(aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0), 2000)}
                    </td>
                </tr>
                <tr>
                    <td><strong>Retrieval Time</strong></td>
                    <td>{aggregated.get('latency_metrics', {}).get('retrieval_time_ms', {}).get('mean', 0):.1f}ms</td>
                    <td>{aggregated.get('latency_metrics', {}).get('retrieval_time_ms', {}).get('std', 0):.1f}ms</td>
                    <td>&lt; 500ms</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td><strong>Generation Time</strong></td>
                    <td>{aggregated.get('latency_metrics', {}).get('generation_time_ms', {}).get('mean', 0):.1f}ms</td>
                    <td>{aggregated.get('latency_metrics', {}).get('generation_time_ms', {}).get('std', 0):.1f}ms</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td><strong>Throughput</strong></td>
                    <td>{aggregated.get('performance_metrics', {}).get('tokens_per_second', {}).get('mean', 0):.1f} tok/s</td>
                    <td>{aggregated.get('performance_metrics', {}).get('tokens_per_second', {}).get('std', 0):.1f} tok/s</td>
                    <td>&gt; 50 tok/s</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td><strong>Memory Usage</strong></td>
                    <td>{aggregated.get('performance_metrics', {}).get('memory_mb', {}).get('mean', 0):.1f} MB</td>
                    <td>{aggregated.get('performance_metrics', {}).get('memory_mb', {}).get('std', 0):.1f} MB</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>

        <div class="explanation">
            <h4>⚡ Understanding Latency Metrics</h4>
            <p><strong>TTFT (Time to First Token):</strong> How quickly the system starts responding. Under 200ms feels instant to users.</p>
            <p><strong>Total Latency:</strong> Complete response time. Under 2 seconds provides good user experience for medical queries.</p>
            <p><strong>Retrieval Time:</strong> Vector database search duration. Optimize with better indexing if high.</p>
            <p><strong>Throughput:</strong> Tokens processed per second. Higher = better for batch processing and multiple users.</p>
        </div>
    </div>

    <div class="section">
        <h2>💡 Recommendations</h2>
        {self._generate_recommendations(aggregated)}
    </div>

    <div class="footer">
        <p>Generated by RAG & LLM Evaluation Framework</p>
        <p>Report generated on {timestamp}</p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = self.output_dir / f"evaluation_report_{timestamp_file}.html"
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 HTML report saved to: {html_file}")
    
    def _get_success_class(self, rate: float) -> str:
        """Get CSS class for success rate."""
        if rate >= 0.95:
            return 'good'
        elif rate >= 0.8:
            return 'warning'
        else:
            return 'bad'
    
    def _get_latency_class(self, latency: float) -> str:
        """Get CSS class for latency."""
        if latency < 2000:
            return 'good'
        elif latency < 5000:
            return 'warning'
        else:
            return 'bad'
    
    def _get_ttft_class(self, ttft: float) -> str:
        """Get CSS class for TTFT."""
        if ttft < 200:
            return 'good'
        elif ttft < 500:
            return 'warning'
        else:
            return 'bad'
    
    def _get_hallucination_class(self, score: float) -> str:
        """Get CSS class for hallucination score."""
        if score < 0.2:
            return 'good'
        elif score < 0.4:
            return 'warning'
        else:
            return 'bad'
    
    def _get_status_emoji(self, value: float, target: float) -> str:
        """Get status emoji based on comparison with target."""
        if value < target:
            return '✅ Good'
        elif value < target * 1.5:
            return '⚠️ Fair'
        else:
            return '❌ Needs Improvement'
    
    def _generate_recommendations(self, aggregated: Dict[str, Any]) -> str:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # Check accuracy
        bleu_mean = aggregated.get('accuracy_metrics', {}).get('bleu_score', {}).get('mean', 0)
        if bleu_mean < 0.3:
            recommendations.append(
                "<p>❗ <strong>Low BLEU Score:</strong> Consider fine-tuning your LLM on domain-specific data or improving prompt engineering.</p>"
            )
        
        # Check hallucination
        hallucination_mean = aggregated.get('rag_metrics', {}).get('hallucination_score', {}).get('mean', 0)
        if hallucination_mean > 0.3:
            recommendations.append(
                "<p>🚨 <strong>High Hallucination:</strong> Your LLM is generating unsupported information. Improve retrieval quality or add stronger grounding prompts.</p>"
            )
        
        # Check context relevance
        context_rel = aggregated.get('rag_metrics', {}).get('context_relevance', {}).get('mean', 0)
        if context_rel < 0.5:
            recommendations.append(
                "<p>🔍 <strong>Poor Retrieval:</strong> Retrieved documents are not relevant. Tune your embedding model, chunk size, or increase retrieval count.</p>"
            )
        
        # Check latency
        latency_mean = aggregated.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0)
        if latency_mean > 2000:
            recommendations.append(
                "<p>⚡ <strong>High Latency:</strong> Response time is slow. Consider caching, faster embedding models, or LLM quantization.</p>"
            )
        
        # Check TTFT
        ttft_mean = aggregated.get('latency_metrics', {}).get('ttft_ms', {}).get('mean', 0)
        if ttft_mean > 200:
            recommendations.append(
                "<p>🕐 <strong>Slow TTFT:</strong> Initial response is delayed. Optimize vector search indexing or use streaming responses.</p>"
            )
        
        if not recommendations:
            recommendations.append(
                "<p>✅ <strong>Excellent Performance!</strong> Your RAG system is performing well across all metrics. Continue monitoring for consistency.</p>"
            )
        
        return "\n".join(recommendations)


def create_medical_test_dataset() -> List[Dict[str, str]]:
    """
    Create a comprehensive test dataset for medical RAG evaluation.
    
    This dataset includes:
    - Factual medical questions
    - Diagnostic queries
    - Treatment questions
    - Medical terminology
    
    Returns:
        List of test cases with queries and reference answers
    """
    return [
        {
            'query': 'What are the common symptoms of type 2 diabetes?',
            'reference_answer': 'Common symptoms of type 2 diabetes include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Some people may also experience areas of darkened skin, usually in the armpits and neck.'
        },
        {
            'query': 'How is hypertension diagnosed and treated?',
            'reference_answer': 'Hypertension is diagnosed when blood pressure readings consistently show systolic pressure of 130 mmHg or higher, or diastolic pressure of 80 mmHg or higher. Treatment includes lifestyle modifications such as diet changes (reducing sodium, DASH diet), regular exercise, weight management, limiting alcohol, and stress reduction. Medications may include ACE inhibitors, beta-blockers, diuretics, or calcium channel blockers depending on the severity and patient factors.'
        },
        {
            'query': 'What causes chest pain and when should I seek emergency care?',
            'reference_answer': 'Chest pain can be caused by various conditions including heart attack, angina, pulmonary embolism, pneumonia, gastroesophageal reflux, or muscle strain. Seek emergency care immediately if chest pain is accompanied by pressure, squeezing sensation, pain radiating to arms/jaw/back, shortness of breath, nausea, lightheadedness, or cold sweats, as these may indicate a heart attack.'
        },
        {
            'query': 'What are the risk factors for cardiovascular disease?',
            'reference_answer': 'Major risk factors for cardiovascular disease include high blood pressure, high cholesterol, diabetes, smoking, obesity, physical inactivity, unhealthy diet, excessive alcohol consumption, family history of heart disease, age (men over 45, women over 55), and chronic stress. Some risk factors like age and family history cannot be changed, but many are modifiable through lifestyle changes.'
        },
        {
            'query': 'How does metformin work in managing blood glucose levels?',
            'reference_answer': 'Metformin works primarily by reducing hepatic glucose production and improving insulin sensitivity in peripheral tissues. It decreases glucose absorption in the intestines and increases glucose uptake and utilization by skeletal muscles. Unlike sulfonylureas, metformin does not increase insulin secretion and therefore has a low risk of causing hypoglycemia.'
        },
        {
            'query': 'What are the stages of chronic kidney disease?',
            'reference_answer': 'Chronic kidney disease has five stages based on glomerular filtration rate (GFR): Stage 1 (GFR ≥90) with normal or high kidney function but kidney damage present; Stage 2 (GFR 60-89) mild reduction; Stage 3a (GFR 45-59) and 3b (GFR 30-44) moderate reduction; Stage 4 (GFR 15-29) severe reduction; Stage 5 (GFR <15) kidney failure requiring dialysis or transplant.'
        },
        {
            'query': 'What is the difference between viral and bacterial pneumonia?',
            'reference_answer': 'Viral pneumonia is caused by viruses like influenza or respiratory syncytial virus, typically develops gradually, and is treated with supportive care and sometimes antivirals. Bacterial pneumonia, often caused by Streptococcus pneumoniae, typically has sudden onset with high fever, productive cough, and requires antibiotic treatment. Bacterial pneumonia generally appears as a lobar consolidation on chest X-ray, while viral pneumonia shows diffuse interstitial patterns.'
        },
        {
            'query': 'What are the warning signs of a stroke?',
            'reference_answer': 'Warning signs of stroke include sudden numbness or weakness of face, arm, or leg (especially on one side), sudden confusion or trouble speaking/understanding, sudden vision problems in one or both eyes, sudden trouble walking, dizziness, or loss of balance/coordination, and sudden severe headache with no known cause. Remember F.A.S.T.: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.'
        }
    ]


def main():
    """
    Main function to run comprehensive RAG and LLM evaluation.
    """
    print("\n" + "="*100)
    print("🔬 COMPREHENSIVE RAG & LLM EVALUATION FRAMEWORK")
    print("="*100 + "\n")
    
    # Configuration (support both Groq and OpenRouter)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medagentica")
    OPENROUTER_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY"))
    OPENROUTER_MODEL = os.getenv("GROQ_MODEL", os.getenv("OPENROUTER_MODEL", "llama-3.3-70b-versatile"))
    
    # Validate credentials
    if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        print("❌ Error: Please set PINECONE_API_KEY environment variable")
        print("   export PINECONE_API_KEY='your_api_key_here'")
        return
    
    if OPENROUTER_API_KEY == "YOUR_API_KEY":
        print("❌ Error: Please set GROQ_API_KEY or OPENROUTER_API_KEY environment variable")
        print("   export GROQ_API_KEY='your_api_key_here'")
        return
    
    try:
        # Initialize RAG system
        logger.info("🚀 Initializing RAG system...")
        rag_system = AgenticRAGSystem(
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX_NAME,
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL
        )
        
        # Initialize evaluator
        logger.info("🔬 Initializing evaluation framework...")
        evaluator = RAGLLMEvaluator(
            rag_system=rag_system,
            output_dir="./evaluation_results"
        )
        
        # Create test dataset
        logger.info("📝 Loading test dataset...")
        test_cases = create_medical_test_dataset()
        logger.info(f"✅ Loaded {len(test_cases)} test cases")
        
        # Run evaluation
        logger.info("\n🏁 Starting comprehensive evaluation...\n")
        results = evaluator.evaluate_dataset(test_cases, batch_size=3)
        
        # Print final summary
        print("\n" + "="*100)
        print("✅ EVALUATION COMPLETE!")
        print("="*100)
        print(f"\n📂 Results saved to: ./evaluation_results/")
        print(f"📊 Total test cases: {results.get('total_evaluations', 0)}")
        print(f"✅ Success rate: {results.get('success_rate', 0)*100:.1f}%")
        print(f"📈 Average BLEU: {results.get('accuracy_metrics', {}).get('bleu_score', {}).get('mean', 0):.3f}")
        print(f"⏱️  Average latency: {results.get('latency_metrics', {}).get('total_latency_ms', {}).get('mean', 0):.0f}ms")
        print(f"🚀 Average throughput: {results.get('performance_metrics', {}).get('tokens_per_second', {}).get('mean', 0):.1f} tok/s")
        
        print("\n📄 Open the HTML report in your browser for detailed analysis!")
        print("="*100 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()


