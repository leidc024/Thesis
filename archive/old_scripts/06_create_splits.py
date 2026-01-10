"""
Dataset Splitting Script
Creates train/validation/test splits with stratified sampling
Target: 70% train / 15% validation / 15% test
"""

import json
import random
from pathlib import Path
from collections import Counter

def load_candidates():
    """Load the selected sentences from the balanced dataset."""
    # Load the sentences file
    sentences_path = "dataset/processed/filipino_sentences_v1.txt"
    candidates_path = "dataset/raw/candidate_sentences.json"
    
    # Load all sentences
    with open(sentences_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Load candidate details
    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidate_data = json.load(f)
    
    # Create lookup by sentence
    sentence_lookup = {c['sentence']: c for c in candidate_data['candidates']}
    
    # Match sentences with their details
    candidates = []
    for sentence in sentences:
        if sentence in sentence_lookup:
            candidates.append(sentence_lookup[sentence])
        else:
            # Create minimal entry for unmatched sentences
            candidates.append({
                'sentence': sentence,
                'ambiguity_types': ['UNKNOWN'],
            })
    
    print(f"Loaded {len(candidates)} candidate entries")
    return candidates

def determine_ambiguity_type(entry):
    """
    Determine the primary ambiguity type for this sentence.
    Priority: COMBINED > D/R > E/I > O/U > UNKNOWN
    """
    types = entry.get('ambiguity_types', ['UNKNOWN'])
    
    # Check for combined (multiple types)
    if 'COMBINED' in types or len(types) >= 2:
        return "COMBINED"
    elif 'D/R' in types:
        return "D/R"
    elif 'E/I' in types:
        return "E/I"
    elif 'O/U' in types:
        return "O/U"
    else:
        return "UNKNOWN"

def stratified_split(candidates, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split candidates into train/val/test using stratified sampling.
    Maintains ambiguity type distribution in each split.
    """
    # Categorize by ambiguity type
    categorized = {
        "E/I": [],
        "O/U": [],
        "D/R": [],
        "COMBINED": [],
        "UNKNOWN": []
    }
    
    for entry in candidates:
        amb_type = determine_ambiguity_type(entry)
        categorized[amb_type].append(entry)
    
    print("\nCategorization:")
    for amb_type, entries in categorized.items():
        print(f"  {amb_type}: {len(entries)} sentences")
    
    # Split each category
    train_data = []
    val_data = []
    test_data = []
    
    for amb_type, entries in categorized.items():
        # Shuffle for randomness
        random.shuffle(entries)
        
        n = len(entries)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.extend(entries[:train_end])
        val_data.extend(entries[train_end:val_end])
        test_data.extend(entries[val_end:])
    
    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_split(data, split_name):
    """Save a split to JSON file."""
    output_path = f"dataset/splits/{split_name}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} entries to {output_path}")

def analyze_split(data, split_name):
    """Analyze the ambiguity type distribution in a split."""
    types = [determine_ambiguity_type(entry) for entry in data]
    counter = Counter(types)
    
    print(f"\n{split_name.upper()} Split Distribution:")
    for amb_type in ["E/I", "O/U", "D/R", "COMBINED", "UNKNOWN"]:
        count = counter[amb_type]
        percentage = (count / len(data)) * 100 if data else 0
        print(f"  {amb_type}: {count} ({percentage:.1f}%)")

def run_splitting():
    """Main splitting workflow."""
    print("="*70)
    print("DATASET SPLITTING - Stratified Train/Val/Test")
    print("="*70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load data
    candidates = load_candidates()
    
    # Create splits
    print("\nCreating stratified splits (70/15/15)...")
    train_data, val_data, test_data = stratified_split(candidates)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(candidates)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/len(candidates)*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/len(candidates)*100:.1f}%)")
    
    # Analyze distributions
    analyze_split(train_data, "train")
    analyze_split(val_data, "validation")
    analyze_split(test_data, "test")
    
    # Save splits
    print("\nSaving splits...")
    save_split(train_data, "train")
    save_split(val_data, "validation")
    save_split(test_data, "test")
    
    print("\n" + "="*70)
    print("SUCCESS: Dataset splits created!")
    print("="*70)
    print("\nNext steps:")
    print("1. Implement graph model using networkx")
    print("2. Integrate RoBERTa-Tagalog for semantic similarity")
    print("3. Apply Personalized PageRank for disambiguation")

if __name__ == "__main__":
    run_splitting()
