"""
Training and Evaluation Script for Baybayin Disambiguation Model v2
Compares baseline (v1) vs enhanced (v2) model with all features.
"""

import json
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from disambiguation_model_v2 import BaybayinDisambiguatorV2, load_dataset

# Configuration
DATA_DIR = Path("dataset")
SPLITS_DIR = DATA_DIR / "splits"
CANDIDATES_FILE = DATA_DIR / "processed" / "candidates_results_v1.json"
RESULTS_DIR = DATA_DIR / "results"


def load_splits():
    """Load train/validation/test splits."""
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        filepath = SPLITS_DIR / f"{split_name}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            splits[split_name] = json.load(f)
        print(f"[OK] Loaded {split_name}: {len(splits[split_name])} sentences")
    return splits


def create_candidates_for_split(split_data, all_candidates):
    """Match split sentences with their OCR candidates."""
    candidates_lookup = {
        entry['ground_truth'].strip().lower(): entry 
        for entry in all_candidates
    }
    
    matched = []
    unmatched = 0
    
    for item in split_data:
        if isinstance(item, dict):
            sentence = item.get('sentence', item.get('text', '')).strip()
        else:
            sentence = str(item).strip()
        
        key = sentence.lower()
        if key in candidates_lookup:
            entry = candidates_lookup[key].copy()
            matched.append(entry)
        else:
            unmatched += 1
    
    if unmatched > 0:
        print(f"  Warning: {unmatched} sentences not found in candidates file")
    
    return matched


def create_simple_splits(all_candidates, train_ratio=0.7, val_ratio=0.15):
    """Create simple train/val/test splits from candidates."""
    import random
    random.seed(42)
    
    data = all_candidates.copy()
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        'train': data[:train_end],
        'validation': data[train_end:val_end],
        'test': data[val_end:]
    }


def main():
    """Main training and evaluation pipeline."""
    print("=" * 70)
    print("BAYBAYIN DISAMBIGUATION - ENHANCED MODEL v2 EVALUATION")
    print("=" * 70)
    print()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading candidates data...")
    all_candidates = load_dataset(str(CANDIDATES_FILE))
    print(f"[OK] Loaded {len(all_candidates)} total candidates")
    print()
    
    # Load and match splits
    print("Attempting to load existing splits...")
    use_existing_splits = True
    try:
        splits = load_splits()
        train_data = create_candidates_for_split(splits['train'], all_candidates)
        val_data = create_candidates_for_split(splits['validation'], all_candidates)
        test_data = create_candidates_for_split(splits['test'], all_candidates)
        
        total_matched = len(train_data) + len(val_data) + len(test_data)
        if total_matched < len(all_candidates) * 0.5:
            print(f"  Only matched {total_matched}/{len(all_candidates)} - using simple splits")
            use_existing_splits = False
    except Exception as e:
        print(f"  Could not load splits: {e}")
        use_existing_splits = False
    
    if not use_existing_splits:
        print("\nCreating simple 70/15/15 splits from candidates...")
        simple_splits = create_simple_splits(all_candidates)
        train_data = simple_splits['train']
        val_data = simple_splits['validation']
        test_data = simple_splits['test']
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} sentences")
    print(f"  Validation: {len(val_data)} sentences")
    print(f"  Test: {len(test_data)} sentences")
    print()
    
    # Initialize enhanced model
    model = BaybayinDisambiguatorV2()
    print()
    
    # Evaluate on validation set
    print("=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)
    
    if len(val_data) > 0:
        val_metrics, val_results = model.evaluate(val_data, verbose=False)
        print()
        print("Validation Results:")
        print(f"  Total Word Accuracy:     {val_metrics['total_accuracy']:.2%}")
        print(f"  Ambiguous Word Accuracy: {val_metrics['ambiguous_accuracy']:.2%}")
        print(f"  Ambiguous Correct:       {val_metrics['correct_ambiguous']}/{val_metrics['total_ambiguous']}")
    print()
    
    # Evaluate on test set
    print("=" * 70)
    print("TEST SET EVALUATION (Final Results)")
    print("=" * 70)
    
    if len(test_data) > 0:
        test_metrics, test_results = model.evaluate(test_data, verbose=False)
        print()
        print("Test Results:")
        print(f"  Total Word Accuracy:     {test_metrics['total_accuracy']:.2%}")
        print(f"  Ambiguous Word Accuracy: {test_metrics['ambiguous_accuracy']:.2%}")
        print(f"  Total Words:             {test_metrics['total_words']}")
        print(f"  Correct Words:           {test_metrics['correct_words']}")
        print(f"  Total Ambiguous:         {test_metrics['total_ambiguous']}")
        print(f"  Correct Ambiguous:       {test_metrics['correct_ambiguous']}")
        
        # Save results
        results_file = RESULTS_DIR / "test_results_v2.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_version': 'v2_enhanced',
                'features': ['semantic', 'frequency', 'cooccurrence', 'morphology'],
                'weights': model.weights,
                'metrics': test_metrics,
                'detailed_results': test_results[:20]
            }, f, ensure_ascii=False, indent=2)
        print()
        print(f"[OK] Results saved to {results_file}")
    
    # Comparison with baselines
    print()
    print("=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print()
    print("Method                          | Ambiguous Accuracy | Notes")
    print("-" * 70)
    print(f"bAI-bAI WE-Only (baseline)      |       77.46%       | Fast, embeddings only")
    print(f"bAI-bAI LLM (baseline)          |       90.52%       | Slow (~3.28s/sample)")
    print(f"Graph+PageRank v1 (ours)        |       76.60%       | Semantic only")
    print(f"Graph+PageRank v2 (ours)        |       {test_metrics['ambiguous_accuracy']:.2%}       | + freq, cooc, morph")
    print()
    
    improvement = test_metrics['ambiguous_accuracy'] - 0.7660
    print(f"Improvement over v1: {improvement:+.2%}")
    
    vs_baseline = test_metrics['ambiguous_accuracy'] - 0.7746
    if vs_baseline > 0:
        print(f"[BETTER] vs WE-Only baseline: {vs_baseline:+.2%}")
    else:
        print(f"[NEEDS WORK] vs WE-Only baseline: {vs_baseline:+.2%}")
    
    # Show sample predictions with feature breakdown
    print()
    print("=" * 70)
    print("SAMPLE PREDICTIONS WITH FEATURE ANALYSIS")
    print("=" * 70)
    
    sample_data = test_data[:3] if len(test_data) >= 3 else all_candidates[:3]
    
    for entry in sample_data:
        print(f"\nGround Truth: {entry['ground_truth']}")
        predicted, debug = model.disambiguate_sentence(
            entry['ocr_candidates'],
            entry['ground_truth']
        )
        print(f"Predicted:    {' '.join(predicted)}")
        
        gt_words = entry['ground_truth'].lower().split()
        for pos, features in debug.get('features', {}).items():
            gt_word = gt_words[pos] if pos < len(gt_words) else "?"
            selected = debug['selected'][pos]
            correct = "[OK]" if selected.lower() == gt_word.lower() else "[X]"
            
            print(f"\n  {correct} Position {pos} (GT: {gt_word}):")
            for candidate, scores in features.items():
                marker = " <--" if candidate == selected else ""
                print(f"      {candidate}: comb={scores['combined']:.3f} "
                      f"(sem={scores['semantic']:.2f}, freq={scores['frequency']:.2f}, "
                      f"cooc={scores['cooccurrence']:.2f}, morph={scores['morphology']:.2f}){marker}")


if __name__ == "__main__":
    main()
