#!/usr/bin/env python3
"""
Evaluation Script for Baybayin Disambiguation Model
Runs clean evaluation with test sentences excluded from corpus.

Usage:
    python evaluate.py                    # Full evaluation
    python evaluate.py --baselines        # Include baseline comparisons
    python evaluate.py --llm gemini       # Test with LLM (gemini, openai, or ollama)
    python evaluate.py --llm-limit 50     # Limit LLM tests (saves API calls)
    python evaluate.py --weights 0.3 0.4 0.2 0.1  # Custom weights
"""

import json
import sys
import io
import argparse
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.disambiguator import BaybayinDisambiguator
from src.baselines import MaBaybayDefault, EmbeddingOnly

# Configuration
TEST_SENTENCES_FILE = "dataset/processed/test_sentences_500.txt"
TEST_CANDIDATES_FILE = "dataset/processed/candidates_results_v2.json"
CORPUS_FILES = [
    "Tagalog_Literary_Text.txt",
    "Tagalog_Religious_Text.txt"
]
OUTPUT_DIR = Path("dataset/results")


def load_test_data():
    """Load test sentences and candidates."""
    with open(TEST_SENTENCES_FILE, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    with open(TEST_CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    return sentences, candidates


def print_results(name: str, metrics: dict):
    """Print formatted results."""
    print(f"\n{name}")
    print("-" * 50)
    print(f"  Total Word Accuracy:     {metrics['total_accuracy']:.2%}")
    print(f"  Ambiguous Word Accuracy: {metrics['ambiguous_accuracy']:.2%}")
    print(f"  Correct Ambiguous:       {metrics['correct_ambiguous']}/{metrics['total_ambiguous']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baybayin Disambiguation Model')
    parser.add_argument('--baselines', action='store_true', help='Include baseline comparisons')
    parser.add_argument('--llm', type=str, choices=['gemini', 'openai', 'ollama'],
                       help='Test with LLM (requires API key or Ollama)')
    parser.add_argument('--llm-limit', type=int, default=None,
                       help='Limit number of sentences for LLM (saves API calls)')
    parser.add_argument('--weights', nargs=4, type=float, 
                       metavar=('SEM', 'FREQ', 'COOC', 'MORPH'),
                       help='Feature weights: semantic frequency cooccurrence morphology')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BAYBAYIN DISAMBIGUATION - EVALUATION")
    print("=" * 70)
    
    # Load test data
    print("\nLoading test data...")
    test_sentences, test_data = load_test_data()
    print(f"  [OK] {len(test_sentences)} test sentences")
    print(f"  [OK] {len(test_data)} candidate entries")
    
    results_all = {}
    
    # Run baselines if requested
    if args.baselines:
        print("\n" + "=" * 70)
        print("BASELINE MODELS")
        print("=" * 70)
        
        # MaBaybay Default
        default_model = MaBaybayDefault()
        metrics_default, _ = default_model.evaluate(test_data)
        results_all['mabaybay_default'] = metrics_default
        print_results("MaBaybay Default (First Candidate)", metrics_default)
        
        # Embedding-Only
        emb_model = EmbeddingOnly()
        metrics_emb, _ = emb_model.evaluate(test_data)
        results_all['embedding_only'] = metrics_emb
        print_results("Embedding-Only (WE-Only)", metrics_emb)
    
    # LLM baseline if requested
    if args.llm:
        print("\n" + "=" * 70)
        print(f"LLM BASELINE ({args.llm.upper()})")
        print("=" * 70)
        
        try:
            from src.baselines import LLMBaseline
            llm_model = LLMBaseline(provider=args.llm)
            
            # Use limit if specified (saves API calls)
            llm_data = test_data[:args.llm_limit] if args.llm_limit else test_data
            if args.llm_limit:
                print(f"  [INFO] Limiting to {args.llm_limit} sentences")
            
            metrics_llm, _ = llm_model.evaluate(llm_data)
            results_all['llm'] = metrics_llm
            print_results(f"LLM ({args.llm} - {llm_model.model})", metrics_llm)
        except Exception as e:
            print(f"  [ERROR] Failed to run LLM baseline: {e}")
            print(f"  [INFO] Make sure you have set the API key environment variable")
            if args.llm == 'gemini':
                print(f"         Set GOOGLE_API_KEY=your_key")
            elif args.llm == 'openai':
                print(f"         Set OPENAI_API_KEY=your_key")
            elif args.llm == 'ollama':
                print(f"         Make sure Ollama is running: ollama serve")
    
    # Main model evaluation
    print("\n" + "=" * 70)
    print("GRAPH-BASED MODEL (Ours)")
    print("=" * 70)
    
    # Set weights
    if args.weights:
        weights = {
            'semantic': args.weights[0],
            'frequency': args.weights[1],
            'cooccurrence': args.weights[2],
            'morphology': args.weights[3]
        }
    else:
        weights = None  # Use defaults
    
    # Initialize model (excluding test sentences from corpus)
    model = BaybayinDisambiguator(
        corpus_files=CORPUS_FILES,
        exclude_sentences=test_sentences,
        weights=weights
    )
    
    # Evaluate
    metrics, detailed_results = model.evaluate(test_data)
    results_all['graph_based'] = metrics
    print_results("Graph-Based + Features (Clean Evaluation)", metrics)
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print()
    print(f"{'Method':<45} | {'Ambiguous Accuracy':>18}")
    print("-" * 70)
    
    if args.baselines:
        print(f"{'MaBaybay Default (no disambiguation)':<45} | {results_all['mabaybay_default']['ambiguous_accuracy']:>17.2%}")
        print(f"{'Embedding-Only (WE-Only)':<45} | {results_all['embedding_only']['ambiguous_accuracy']:>17.2%}")
    
    if 'llm' in results_all:
        print(f"{'LLM (' + args.llm + ') - Our Test':<45} | {results_all['llm']['ambiguous_accuracy']:>17.2%}")
    
    print(f"{'bAI-bAI WE-Only (reported in paper)':<45} | {'77.46%':>18}")
    print(f"{'bAI-bAI LLM (reported in paper)':<45} | {'90.52%':>18}")
    print(f"{'Our Graph+Features (Clean Evaluation)':<45} | {metrics['ambiguous_accuracy']:>17.2%}")
    
    # Improvement analysis
    improvement_we = metrics['ambiguous_accuracy'] - 0.7746
    improvement_llm = metrics['ambiguous_accuracy'] - 0.9052
    print()
    print(f"Improvement over WE-Only:  {improvement_we:+.2%} ({'+' if improvement_we > 0 else ''}{improvement_we*100:.1f} percentage points)")
    print(f"Improvement over LLM:      {improvement_llm:+.2%} ({'+' if improvement_llm > 0 else ''}{improvement_llm*100:.1f} percentage points)")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "evaluation_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'test_sentences': len(test_sentences),
                'weights': model.weights
            },
            'results': {k: {kk: float(vv) if isinstance(vv, float) else vv 
                          for kk, vv in v.items()} 
                       for k, v in results_all.items()}
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_file}")


if __name__ == "__main__":
    main()
