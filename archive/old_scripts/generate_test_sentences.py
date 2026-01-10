"""
Generate 500 Test Sentences for Baybayin Disambiguation
Creates a clean test set that should NOT be used for frequency statistics.

Distribution target:
- 37.5% E/I ambiguity (188 sentences)
- 37.5% O/U ambiguity (188 sentences)  
- 15% D/R ambiguity (75 sentences)
- 10% COMBINED ambiguity (49 sentences)
"""

import re
import json
import random
from pathlib import Path
from collections import defaultdict

# Configuration
LITERARY_CORPUS = "Tagalog_Literary_Text.txt"
RELIGIOUS_CORPUS = "Tagalog_Religious_Text.txt"
AMBIGUOUS_PAIRS_JSON = "dataset/analysis/ambiguous_pairs_complete.json"
OUTPUT_FILE = "dataset/processed/test_sentences_500.txt"
OUTPUT_JSON = "dataset/processed/test_sentences_500.json"

MIN_WORDS = 5
MAX_WORDS = 12

# Target distribution
TARGET_TOTAL = 500
DISTRIBUTION = {
    'E/I': 0.375,      # 188 sentences
    'O/U': 0.375,      # 188 sentences
    'D/R': 0.15,       # 75 sentences
    'COMBINED': 0.10   # 49 sentences
}


def load_ambiguous_pairs():
    """Load ambiguous pairs data."""
    with open(AMBIGUOUS_PAIRS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_words = set()
    word_to_type = {}
    word_to_group = {}
    
    for group in data['ambiguous_groups']:
        amb_type = group['ambiguity_type']
        words = [w.lower() for w in group['words']]
        
        for word in words:
            all_words.add(word)
            word_to_type[word] = amb_type
            word_to_group[word] = words
    
    print(f"[OK] Loaded {len(all_words)} ambiguous words")
    print(f"  E/I: {data['statistics']['by_type'].get('E/I', 0)} patterns")
    print(f"  O/U: {data['statistics']['by_type'].get('O/U', 0)} patterns")
    print(f"  D/R: {data['statistics']['by_type'].get('D/R', 0)} patterns")
    print(f"  COMBINED: {data['statistics']['by_type'].get('COMBINED', 0)} patterns")
    
    return all_words, word_to_type, word_to_group


def is_clean_sentence(sentence):
    """Check if sentence is clean for OCR testing."""
    # Reject complex punctuation
    if any(p in sentence for p in ['...', '—', '–', '"', '"', ';', ':', '(', ')', '[', ']']):
        return False
    
    # Reject HTML tags
    if '<' in sentence or '>' in sentence:
        return False
    
    # Must start with capital
    if not sentence or not sentence[0].isupper():
        return False
    
    # No dialogue fragments
    if sentence.startswith('"') or sentence.startswith("'"):
        return False
    
    # No numbers
    if re.search(r'\d', sentence):
        return False
    
    # Check quote balance
    for q in ['"', "'"]:
        if sentence.count(q) % 2 != 0:
            return False
    
    return True


def clean_sentence(sentence):
    """Clean sentence for OCR."""
    sentence = sentence.replace('-', ' ')
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


def extract_sentences(text):
    """Extract sentences from text."""
    sentences = re.split(r'[.!?]+\s+', text)
    
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        sent = re.sub(r'\s+', ' ', sent)
        
        if len(sent) < 15:
            continue
        
        if not is_clean_sentence(sent):
            continue
        
        sent = clean_sentence(sent)
        words = sent.split()
        
        if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
            continue
        
        cleaned.append(sent)
    
    return cleaned


def classify_sentence(sentence, all_words, word_to_type):
    """
    Classify sentence by its primary ambiguity type.
    Returns (type, list of ambiguous words found).
    """
    words = [w.lower().strip('.,!?') for w in sentence.split()]
    
    found = []
    types_found = defaultdict(int)
    
    for word in words:
        if word in all_words:
            amb_type = word_to_type[word]
            found.append((word, amb_type))
            types_found[amb_type] += 1
    
    if not found:
        return None, []
    
    # Primary type is the most common, or COMBINED if multiple types
    if len(types_found) > 1:
        # Check if there's a COMBINED word
        if 'COMBINED' in types_found:
            return 'COMBINED', found
        # Multiple different types in one sentence - could classify as complex
        # For simplicity, use the dominant type
        primary = max(types_found.keys(), key=lambda k: types_found[k])
    else:
        primary = list(types_found.keys())[0]
    
    return primary, found


def main():
    print("=" * 60)
    print("GENERATING 500 TEST SENTENCES")
    print("=" * 60)
    print()
    
    # Load ambiguous pairs
    all_words, word_to_type, word_to_group = load_ambiguous_pairs()
    print()
    
    # Load text corpora
    print("Loading text corpora...")
    all_sentences = []
    
    for corpus_file in [LITERARY_CORPUS, RELIGIOUS_CORPUS]:
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                text = f.read()
            sentences = extract_sentences(text)
            print(f"  {corpus_file}: {len(sentences)} clean sentences")
            all_sentences.extend([(s, corpus_file) for s in sentences])
        except FileNotFoundError:
            print(f"  Warning: {corpus_file} not found")
    
    print(f"Total candidate sentences: {len(all_sentences)}")
    print()
    
    # Classify sentences by ambiguity type
    print("Classifying sentences by ambiguity type...")
    by_type = defaultdict(list)
    
    for sentence, source in all_sentences:
        amb_type, found_words = classify_sentence(sentence, all_words, word_to_type)
        if amb_type:
            by_type[amb_type].append({
                'sentence': sentence,
                'source': source,
                'ambiguous_words': found_words
            })
    
    for amb_type, sentences in by_type.items():
        print(f"  {amb_type}: {len(sentences)} sentences available")
    print()
    
    # Calculate targets
    targets = {
        'E/I': int(TARGET_TOTAL * DISTRIBUTION['E/I']),
        'O/U': int(TARGET_TOTAL * DISTRIBUTION['O/U']),
        'D/R': int(TARGET_TOTAL * DISTRIBUTION['D/R']),
        'COMBINED': int(TARGET_TOTAL * DISTRIBUTION['COMBINED'])
    }
    
    # Adjust to exactly 500
    total = sum(targets.values())
    if total < TARGET_TOTAL:
        targets['E/I'] += TARGET_TOTAL - total
    
    print("Target distribution:")
    for amb_type, count in targets.items():
        print(f"  {amb_type}: {count}")
    print()
    
    # Sample sentences
    print("Sampling sentences...")
    selected = []
    selected_texts = set()  # Avoid duplicates
    
    for amb_type, target_count in targets.items():
        available = by_type[amb_type]
        random.shuffle(available)
        
        count = 0
        for item in available:
            if count >= target_count:
                break
            
            # Skip duplicates
            if item['sentence'].lower() in selected_texts:
                continue
            
            selected.append({
                'sentence': item['sentence'],
                'source': 'Literary' if 'Literary' in item['source'] else 'Religious',
                'primary_type': amb_type,
                'ambiguous_words': item['ambiguous_words']
            })
            selected_texts.add(item['sentence'].lower())
            count += 1
        
        print(f"  {amb_type}: selected {count}/{target_count}")
    
    print(f"\nTotal selected: {len(selected)}")
    
    # Shuffle final list
    random.shuffle(selected)
    
    # Save as plain text (just sentences)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in selected:
            f.write(item['sentence'] + '\n')
    print(f"\n[OK] Saved sentences to {OUTPUT_FILE}")
    
    # Save as JSON with metadata
    output_data = {
        'metadata': {
            'total_sentences': len(selected),
            'distribution': {
                'E/I': sum(1 for s in selected if s['primary_type'] == 'E/I'),
                'O/U': sum(1 for s in selected if s['primary_type'] == 'O/U'),
                'D/R': sum(1 for s in selected if s['primary_type'] == 'D/R'),
                'COMBINED': sum(1 for s in selected if s['primary_type'] == 'COMBINED')
            },
            'note': 'These sentences should NOT be used for corpus frequency statistics to avoid data leakage'
        },
        'sentences': selected
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved metadata to {OUTPUT_JSON}")
    
    # Print final distribution
    print("\n" + "=" * 60)
    print("FINAL DISTRIBUTION")
    print("=" * 60)
    for amb_type in ['E/I', 'O/U', 'D/R', 'COMBINED']:
        count = output_data['metadata']['distribution'][amb_type]
        pct = count / len(selected) * 100
        print(f"  {amb_type}: {count} ({pct:.1f}%)")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES")
    print("=" * 60)
    for amb_type in ['E/I', 'O/U', 'D/R', 'COMBINED']:
        samples = [s for s in selected if s['primary_type'] == amb_type][:2]
        print(f"\n{amb_type}:")
        for s in samples:
            words = [f"'{w[0]}'" for w in s['ambiguous_words']]
            print(f"  \"{s['sentence']}\"")
            print(f"    Ambiguous: {', '.join(words)}")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
