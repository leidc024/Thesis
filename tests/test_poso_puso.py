"""
Test disambiguator on poso/puso ambiguous pair
Compares Context-Aware Baybayin Transliteration vs MaBaybay Default (First Candidate)
Testing with 100 sentences (50 each)
"""

import json
import re
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path (so tests/ can import src/)
PROJECT_ROOT = str(Path(__file__).parent.parent)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.disambiguator import BaybayinDisambiguator

def get_clean_words(sentence):
    """Extract words from sentence, removing punctuation"""
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', sentence.lower())
    return words

# Read the sentences from gold standard dataset
SENTENCE_FILE = "gold_standard_dataset/sentences/12_poso_puso.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with poso vs puso are mixed throughout"""
    poso_sentences = []
    puso_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "poso" in words:
            poso_sentences.append(line)
        elif "puso" in words:
            puso_sentences.append(line)
    
    return poso_sentences, puso_sentences

poso_sentences, puso_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("POSO/PUSO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nPoso sentences: {len(poso_sentences)}")
print(f"Puso sentences: {len(puso_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

poso_with_target = []
poso_without_target = []
puso_with_target = []
puso_without_target = []

for i, sent in enumerate(poso_sentences, 1):
    words = get_clean_words(sent)
    if "poso" in words:
        poso_with_target.append((i, sent))
    else:
        poso_without_target.append((i, sent))

for i, sent in enumerate(puso_sentences, 1):
    words = get_clean_words(sent)
    if "puso" in words:
        puso_with_target.append((i+len(poso_sentences), sent))
    else:
        puso_without_target.append((i+len(poso_sentences), sent))

print(f"\nPOSO sentences with 'poso': {len(poso_with_target)}/{len(poso_sentences)}")
print(f"PUSO sentences with 'puso': {len(puso_with_target)}/{len(puso_sentences)}")

if poso_without_target:
    print(f"\n⚠️  POSO sentences WITHOUT 'poso' word ({len(poso_without_target)}):")
    for line_num, sent in poso_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(poso_without_target) > 5:
        print(f"  ... and {len(poso_without_target) - 5} more")

if puso_without_target:
    print(f"\n⚠️  PUSO sentences WITHOUT 'puso' word ({len(puso_without_target)}):")
    for line_num, sent in puso_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(puso_without_target) > 5:
        print(f"  ... and {len(puso_without_target) - 5} more")

# Create test data with OCR candidates
# For poso/puso, both map to Baybayin ᜉᜓᜐᜓ
# MaBaybay default order: ["poso", "puso"] (poso is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add poso sentences (ground truth = poso)
for sent in poso_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "poso":
            # Ambiguous position - both candidates (MaBaybay order: poso first)
            candidates.append(["poso", "puso"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add puso sentences (ground truth = puso)
for sent in puso_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "puso":
            # Ambiguous position - both candidates (MaBaybay order: poso first)
            candidates.append(["poso", "puso"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

print(f"\nTotal test cases: {len(test_data)}")

# ============================================================================
# BASELINE: MaBaybay Default (First Candidate Selection)
# ============================================================================
print("\n" + "="*70)
print("BASELINE: MaBaybay Default (Always Pick First Candidate)")
print("="*70)

# First candidate is always "poso" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_poso = 0
baseline_correct_puso = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "poso" (first candidate in MaBaybay order)
    if "poso" in gt_words:
        baseline_correct_total += 1
        baseline_correct_poso += 1
    # If ground truth is "puso", baseline gets it wrong (picks "poso")
    # So baseline_correct_puso stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'poso' (first candidate)")
print(f"Poso accuracy: {baseline_correct_poso}/50 = {baseline_correct_poso/50:.2%}")
print(f"Puso accuracy: {baseline_correct_puso}/50 = {baseline_correct_puso/50:.2%}")
print(f"Overall baseline accuracy: {baseline_correct_total}/100 = {baseline_accuracy:.2f}%")

# ============================================================================
# INITIALIZE MODEL (shared across all methods)
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

all_test_sentences = [item['ground_truth'] for item in test_data]
model = BaybayinDisambiguator(
    corpus_files=[
        "corpus/Tagalog_Literary_Text.txt",
        "corpus/Tagalog_Religious_Text.txt",
        "corpus/Tagalog_Balita_Texts_Balanced.txt"
    ],
    exclude_sentences=all_test_sentences  # Clean evaluation - no data leakage
)

# ============================================================================
# METHOD 1: Pure MLM-PLL (Semantic Only)
# ============================================================================

mlm_only_weights = {
    'semantic': 1.0,
    'frequency': 0.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}
print(f"\nWeights: {mlm_only_weights}")
print("Semantic scoring: MLM PLL (Masked Language Model Pseudo-Log-Likelihood)")
print("Running evaluation...")

mlm_only_metrics, mlm_only_results = model.evaluate(
    test_data, show_progress=True, use_mlm=True, weights_override=mlm_only_weights
)
mlm_only_accuracy = mlm_only_metrics['ambiguous_accuracy'] * 100
print(f"★ Pure MLM-PLL accuracy: {mlm_only_accuracy:.2f}%")

# Use MLM-PLL results for detailed predictions display
metrics = mlm_only_metrics
results = mlm_only_results

# Display results
print("\n" + "="*70)
print("CONTEXT-AWARE DISAMBIGUATION RESULTS")
print("="*70)

print(f"\nAmbiguous words (poso/puso): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_poso_examples = []
incorrect_poso_examples = []
correct_puso_examples = []
incorrect_puso_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a poso or puso sentence
    if "poso" in gt_words:
        if "poso" in pred_words:
            correct_poso_examples.append((i+1, gt, pred))
        else:
            incorrect_poso_examples.append((i+1, gt, pred))
    elif "puso" in gt_words:
        if "puso" in pred_words:
            correct_puso_examples.append((i+1, gt, pred))
        else:
            incorrect_puso_examples.append((i+1, gt, pred))

# Display POSO results
print(f"\n{'='*70}")
print(f"POSO SENTENCES: {len(correct_poso_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_poso_examples:
    print(f"\n✓ CORRECT POSO PREDICTIONS ({len(correct_poso_examples)}):")
    for idx, gt, pred in correct_poso_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_poso_examples:
    print(f"\n✗ INCORRECT POSO PREDICTIONS ({len(incorrect_poso_examples)}):")
    for idx, gt, pred in incorrect_poso_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display PUSO results
print(f"\n{'='*70}")
print(f"PUSO SENTENCES: {len(correct_puso_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_puso_examples:
    print(f"\n✓ CORRECT PUSO PREDICTIONS ({len(correct_puso_examples)}):")
    for idx, gt, pred in correct_puso_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_puso_examples:
    print(f"\n✗ INCORRECT PUSO PREDICTIONS ({len(incorrect_puso_examples)}):")
    for idx, gt, pred in incorrect_puso_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
poso_correct = 0
puso_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "poso" in gt_words:
        if "poso" in pred_words:
            poso_correct += 1
    elif "puso" in gt_words:
        if "puso" in pred_words:
            puso_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD (MLM Method)")
print("="*70)
print(f"\nPoso accuracy: {poso_correct}/50 = {poso_correct/50:.2%}")
print(f"Puso accuracy: {puso_correct}/50 = {puso_correct/50:.2%}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

context_accuracy = mlm_only_accuracy
improvement = context_accuracy - baseline_accuracy

# Count per-word accuracy for Pure MLM-PLL method
def count_word_accuracy(result_list, word1, word2):
    w1_correct = 0
    w2_correct = 0
    for test_item, result_item in zip(test_data, result_list):
        gt_words = get_clean_words(test_item['ground_truth'])
        pred_words = get_clean_words(result_item['predicted'])
        if word1 in gt_words:
            if word1 in pred_words:
                w1_correct += 1
        elif word2 in gt_words:
            if word2 in pred_words:
                w2_correct += 1
    return w1_correct, w2_correct

mlm_poso, mlm_puso = count_word_accuracy(mlm_only_results, "poso", "puso")

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │{{{word1.capitalize()}}} (50) │ {{{word2.capitalize()}}} (50) │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {{baseline_accuracy:6.2f}}%    │   {{baseline_correct_{word1}:2d}}/50    │    {{baseline_correct_{word2}:2d}}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ Pure MLM-PLL               │   {{context_accuracy:6.2f}}%    │   {{mlm_{word1}:2d}}/50    │    {{mlm_{word2}:2d}}/50     │
│   (Semantic Only)            │ ({{improvement:+6.2f}}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      Baseline always picks first candidate.
""")

# Save detailed results
output = {
    'ambiguous_pair': 'poso, puso',
    'baybayin': 'ᜉᜓᜐᜓ',
    'type': 'O/U',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'poso_accuracy': f"{baseline_correct_poso}/50",
            'puso_accuracy': f"{baseline_correct_puso}/50"
        },
        'cosine_only': {
            'name': 'Pure Cosine Similarity (Semantic Only)',
            'strategy': '100% cosine similarity of mean-pooled RoBERTa embeddings, no other features',
            'accuracy': cosine_only_accuracy,
            'correct': cosine_only_metrics['correct_ambiguous'],
            'poso_accuracy': f"{cosine_only_poso}/50",
            'puso_accuracy': f"{cosine_only_puso}/50"
        },
        'cosine_multi': {
            'name': 'Cosine Similarity + Multi-Feature (Old Method)',
            'strategy': 'Cosine similarity semantic + frequency + cooccurrence + morphology',
            'accuracy': cosine_multi_accuracy,
            'correct': cosine_multi_metrics['correct_ambiguous'],
            'poso_accuracy': f"{cosine_multi_poso}/50",
            'puso_accuracy': f"{cosine_multi_puso}/50"
        },
        'mlm_pll': {
            'name': 'MLM PLL + Multi-Feature (Current)',
            'strategy': 'MLM pseudo-log-likelihood semantic + frequency + cooccurrence + morphology',
            'accuracy': context_accuracy,
            'correct': metrics['correct_ambiguous'],
            'poso_accuracy': f"{mlm_poso}/50",
            'puso_accuracy': f"{mlm_puso}/50"
        },
        'improvement_over_baseline': improvement
    },
    'metrics': mlm_only_metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_poso_puso.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_poso_puso.json")
print("="*70)
