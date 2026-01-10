"""
Training and Evaluation Script for Baybayin Disambiguation Model
Uses the train/validation/test splits from your dataset.
"""

import json
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from disambiguation_model import BaybayinDisambiguator, load_dataset

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
    """
    Match split sentences with their OCR candidates.
    
    Args:
        split_data: List of sentences from split file
        all_candidates: Full candidates results
        
    Returns:
        List of matched entries with candidates
    """
    # Create lookup by ground truth text
    candidates_lookup = {
        entry['ground_truth'].strip().lower(): entry 
        for entry in all_candidates
    }
    
    matched = []
    unmatched = 0
    
    for item in split_data:
        # Handle both dict format and string format
        if isinstance(item, dict):
            sentence = item.get('sentence', item.get('text', '')).strip()
        else:
            sentence = str(item).strip()
        
        # Look up candidates
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
    """
    Create simple train/val/test splits from candidates if splits don't match.
    
    Args:
        all_candidates: All candidate entries
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        dict with train, validation, test splits
    """
    import random
    random.seed(42)  # For reproducibility
    
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
    print("=" * 60)
    print("BAYBAYIN DISAMBIGUATION - TRAINING & EVALUATION")
    print("=" * 60)
    print()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading candidates data...")
    all_candidates = load_dataset(str(CANDIDATES_FILE))
    print(f"[OK] Loaded {len(all_candidates)} total candidates")
    print()
    
    # Try to load and match with existing splits
    print("Attempting to load existing splits...")
    use_existing_splits = True
    try:
        splits = load_splits()
        train_data = create_candidates_for_split(splits['train'], all_candidates)
        val_data = create_candidates_for_split(splits['validation'], all_candidates)
        test_data = create_candidates_for_split(splits['test'], all_candidates)
        
        # Check if we have enough matched data
        total_matched = len(train_data) + len(val_data) + len(test_data)
        if total_matched < len(all_candidates) * 0.5:
            print(f"  Only matched {total_matched}/{len(all_candidates)} - using simple splits instead")
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
    
    # Initialize model
    print("Initializing disambiguation model...")
    model = BaybayinDisambiguator()
    print()
    
    # Note: This model doesn't require traditional "training" since it uses
    # pre-trained RoBERTa + graph-based inference. The train set can be used
    # for hyperparameter tuning (damping factor, similarity threshold, etc.)
    
    # Evaluate on validation set
    print("=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    
    if len(val_data) > 0:
        val_metrics, val_results = model.evaluate(val_data[:50], verbose=False)
        print()
        print("Validation Results:")
        print(f"  Total Word Accuracy: {val_metrics['total_accuracy']:.2%}")
        print(f"  Ambiguous Word Accuracy: {val_metrics['ambiguous_accuracy']:.2%}")
        print(f"  Ambiguous: {val_metrics['correct_ambiguous']}/{val_metrics['total_ambiguous']}")
    else:
        print("No validation data matched!")
    print()
    
    # Evaluate on test set
    print("=" * 60)
    print("TEST SET EVALUATION (Final Results)")
    print("=" * 60)
    
    if len(test_data) > 0:
        test_metrics, test_results = model.evaluate(test_data, verbose=False)
        print()
        print("Test Results:")
        print(f"  Total Word Accuracy: {test_metrics['total_accuracy']:.2%}")
        print(f"  Ambiguous Word Accuracy: {test_metrics['ambiguous_accuracy']:.2%}")
        print(f"  Total Words: {test_metrics['total_words']}")
        print(f"  Correct Words: {test_metrics['correct_words']}")
        print(f"  Total Ambiguous: {test_metrics['total_ambiguous']}")
        print(f"  Correct Ambiguous: {test_metrics['correct_ambiguous']}")
        
        # Save results
        results_file = RESULTS_DIR / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': test_metrics,
                'detailed_results': test_results[:20]  # Save first 20 detailed
            }, f, ensure_ascii=False, indent=2)
        print()
        print(f"[OK] Results saved to {results_file}")
    else:
        print("No test data matched!")
    
    # Show some example predictions
    print()
    print("=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    sample_data = test_data[:5] if len(test_data) >= 5 else all_candidates[:5]
    
    for entry in sample_data:
        print(f"\nGround Truth: {entry['ground_truth']}")
        predicted, debug = model.disambiguate_sentence(
            entry['ocr_candidates'],
            entry['ground_truth']
        )
        print(f"Predicted:    {' '.join(predicted)}")
        
        # Show ambiguous word decisions
        gt_words = entry['ground_truth'].lower().split()
        for pos, selected in debug['selected'].items():
            gt_word = gt_words[pos] if pos < len(gt_words) else "?"
            correct = "[CORRECT]" if selected.lower() == gt_word.lower() else "[WRONG]"
            candidates = entry['ocr_candidates'][pos]
            print(f"  {correct} Position {pos}: {candidates} -> {selected} (GT: {gt_word})")


if __name__ == "__main__":
    main()
