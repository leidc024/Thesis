"""
Quick test: Pure MLM (semantic only) vs MLM + Multi-Feature
Answers the question: do the other 3 features help or hurt when using MLM PLL?
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import re
from src.disambiguator import BaybayinDisambiguator

def get_clean_words(sentence):
    return re.findall(r'\b\w+\b', sentence.lower())

def parse_sentence_file(filepath, word_a, word_b):
    a_sents, b_sents = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        words = get_clean_words(line)
        if word_a in words:
            a_sents.append(line)
        elif word_b in words:
            b_sents.append(line)
    return a_sents, b_sents

def run_test(disambiguator, sentences_a, sentences_b, word_a, word_b, candidates, use_mlm, weights_override=None):
    """Run evaluation and return per-word accuracy."""
    all_sents = sentences_a + sentences_b
    test_data = []
    for sent in all_sents:
        words = get_clean_words(sent)
        ocr = []
        for w in words:
            if w == word_a or w == word_b:
                ocr.append(list(candidates))
            else:
                ocr.append(w)
        test_data.append({'sentence': sent, 'ocr_candidates': ocr, 'ground_truth': sent})
    
    metrics, details = disambiguator.evaluate(test_data, use_mlm=use_mlm, weights_override=weights_override)
    
    a_correct = 0
    b_correct = 0
    n_a = len(sentences_a)
    
    for i, r in enumerate(details):
        predicted = r['predicted'].lower()
        if i < n_a:
            if word_a in get_clean_words(predicted):
                a_correct += 1
        else:
            if word_b in get_clean_words(predicted):
                b_correct += 1
    
    return a_correct, len(sentences_a), b_correct, len(sentences_b)

# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PURE MLM vs MLM + MULTI-FEATURE COMPARISON")
print("=" * 70)

disambiguator = BaybayinDisambiguator()

# Test configurations
pairs = [
    {
        'name': 'bote/buti',
        'file': 'gold_standard_dataset/sentences/02_bola_bula.txt',
        'word_a': 'bola', 'word_b': 'bula',
        'candidates': ["bola", "bula"]
    },
    {
        'name': 'itodo/ituro', 
        'file': 'gold_standard_dataset/sentences/07_itodo_ituro.txt',
        'word_a': 'itodo', 'word_b': 'ituro',
        'candidates': ["itodo", "ituro"]
    },
    {
        'name': 'kompas/kumpas',
        'file': 'gold_standard_dataset/sentences/08_kompas_kumpas.txt',
        'word_a': 'kompas', 'word_b': 'kumpas',
        'candidates': ["kompas", "kumpas"]
    },
]

methods = [
    {
        'name': 'Pure MLM (Semantic Only)',
        'use_mlm': True,
        'weights': {'semantic': 1.0, 'frequency': 0.0, 'cooccurrence': 0.0, 'morphology': 0.0}
    },
    {
        'name': 'MLM + Multi-Feature',
        'use_mlm': True,
        'weights': None  # Use default weights
    },
]

results_table = {}

for pair in pairs:
    print(f"\n{'─' * 60}")
    print(f"  Testing: {pair['name']}")
    print(f"{'─' * 60}")
    
    sents_a, sents_b = parse_sentence_file(pair['file'], pair['word_a'], pair['word_b'])
    results_table[pair['name']] = {}
    
    for method in methods:
        print(f"\n  {method['name']}...")
        a_c, a_t, b_c, b_t = run_test(
            disambiguator, sents_a, sents_b,
            pair['word_a'], pair['word_b'], pair['candidates'],
            use_mlm=method['use_mlm'],
            weights_override=method['weights']
        )
        total = a_c + b_c
        overall = (total / (a_t + b_t)) * 100
        print(f"    {pair['word_a']}: {a_c}/{a_t}, {pair['word_b']}: {b_c}/{b_t}, Overall: {overall:.0f}%")
        results_table[pair['name']][method['name']] = {
            'a': f"{a_c}/{a_t}", 'b': f"{b_c}/{b_t}", 'overall': overall
        }

# ══════════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 70}")
print("  FINAL COMPARISON: Are the other 3 features helping MLM?")
print("=" * 70)

print(f"""
┌──────────────┬───────────────────────┬───────────────────────┬────────────┐
│    Pair       │  Pure MLM             │  MLM + Multi-Feature  │   Effect   │
│              │  (semantic only)      │  (+freq/cooc/morph)   │            │
├──────────────┼───────────────────────┼───────────────────────┼────────────┤""")

for pair_name, data in results_table.items():
    pure = data['Pure MLM (Semantic Only)']
    multi = data['MLM + Multi-Feature']
    diff = multi['overall'] - pure['overall']
    effect = f"+{diff:.0f}%" if diff > 0 else f"{diff:.0f}%"
    symbol = "✅" if diff > 0 else ("─" if diff == 0 else "❌")
    print(f"│ {pair_name:12s} │  {pure['overall']:5.0f}% ({pure['a']:>5s}, {pure['b']:>5s}) │  {multi['overall']:5.0f}% ({multi['a']:>5s}, {multi['b']:>5s}) │ {effect:>5s} {symbol}  │")

print(f"└──────────────┴───────────────────────┴───────────────────────┴────────────┘")

print("""
INTERPRETATION:
  • If the effect is POSITIVE → other features are helping MLM
  • If the effect is NEGATIVE → other features are hurting MLM (frequency bias)
  • If the effect is ~0      → other features are irrelevant with MLM
""")
