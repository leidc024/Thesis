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
# ...existing code...

