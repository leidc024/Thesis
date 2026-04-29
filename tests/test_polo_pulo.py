"""
Test disambiguator on polo/pulo ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/11_polo_pulo.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with polo vs pulo are mixed throughout"""
    polo_sentences = []
    pulo_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "polo" in words:
            polo_sentences.append(line)
        elif "pulo" in words:
            pulo_sentences.append(line)
    
    return polo_sentences, pulo_sentences

polo_sentences, pulo_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("POLO/PULO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nPolo sentences: {len(polo_sentences)}")
print(f"Pulo sentences: {len(pulo_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

polo_with_target = []
polo_without_target = []
pulo_with_target = []
pulo_without_target = []

for i, sent in enumerate(polo_sentences, 1):
    words = get_clean_words(sent)
    if "polo" in words:
        polo_with_target.append((i, sent))
    else:
        polo_without_target.append((i, sent))

for i, sent in enumerate(pulo_sentences, 1):
    words = get_clean_words(sent)
    if "pulo" in words:
        pulo_with_target.append((i+len(polo_sentences), sent))
    else:
        pulo_without_target.append((i+len(polo_sentences), sent))

print(f"\nPOLO sentences with 'polo': {len(polo_with_target)}/{len(polo_sentences)}")
print(f"PULO sentences with 'pulo': {len(pulo_with_target)}/{len(pulo_sentences)}")

if polo_without_target:
    print(f"\n⚠️  POLO sentences WITHOUT 'polo' word ({len(polo_without_target)}):")
    for line_num, sent in polo_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(polo_without_target) > 5:
        print(f"  ... and {len(polo_without_target) - 5} more")

if pulo_without_target:
    print(f"\n⚠️  PULO sentences WITHOUT 'pulo' word ({len(pulo_without_target)}):")
    for line_num, sent in pulo_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(pulo_without_target) > 5:
        print(f"  ... and {len(pulo_without_target) - 5} more")

# Create test data with OCR candidates
# For polo/pulo, both map to Baybayin ᜉᜓᜎᜓ
# MaBaybay default order: ["polo", "pulo"] (polo is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add polo sentences (ground truth = polo)
for sent in polo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "polo":
            # Ambiguous position - both candidates (MaBaybay order: polo first)
            candidates.append(["polo", "pulo"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add pulo sentences (ground truth = pulo)
for sent in pulo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "pulo":
            # Ambiguous position - both candidates (MaBaybay order: polo first)
            candidates.append(["polo", "pulo"])
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

# First candidate is always "polo" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_polo = 0
baseline_correct_pulo = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "polo" (first candidate in MaBaybay order)
    if "polo" in gt_words:
        baseline_correct_total += 1
        baseline_correct_polo += 1
    # If ground truth is "pulo", baseline gets it wrong (picks "polo")
    # So baseline_correct_pulo stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'polo' (first candidate)")
print(f"Polo accuracy: {baseline_correct_polo}/50 = {baseline_correct_polo/50:.2%}")
print(f"Pulo accuracy: {baseline_correct_pulo}/50 = {baseline_correct_pulo/50:.2%}")
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

print(f"\nAmbiguous words (polo/pulo): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_polo_examples = []
incorrect_polo_examples = []
correct_pulo_examples = []
incorrect_pulo_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a polo or pulo sentence
    if "polo" in gt_words:
        if "polo" in pred_words:
            correct_polo_examples.append((i+1, gt, pred))
        else:
            incorrect_polo_examples.append((i+1, gt, pred))
    elif "pulo" in gt_words:
        if "pulo" in pred_words:
            correct_pulo_examples.append((i+1, gt, pred))
        else:
            incorrect_pulo_examples.append((i+1, gt, pred))

# Display POLO results
print(f"\n{'='*70}")
print(f"POLO SENTENCES: {len(correct_polo_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_polo_examples:
    print(f"\n✓ CORRECT POLO PREDICTIONS ({len(correct_polo_examples)}):")
    for idx, gt, pred in correct_polo_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_polo_examples:
    print(f"\n✗ INCORRECT POLO PREDICTIONS ({len(incorrect_polo_examples)}):")
    for idx, gt, pred in incorrect_polo_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display PULO results
print(f"\n{'='*70}")
print(f"PULO SENTENCES: {len(correct_pulo_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_pulo_examples:
    print(f"\n✓ CORRECT PULO PREDICTIONS ({len(correct_pulo_examples)}):")
    for idx, gt, pred in correct_pulo_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_pulo_examples:
    print(f"\n✗ INCORRECT PULO PREDICTIONS ({len(incorrect_pulo_examples)}):")
    for idx, gt, pred in incorrect_pulo_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
polo_correct = 0
pulo_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "polo" in gt_words:
        if "polo" in pred_words:
            polo_correct += 1
    elif "pulo" in gt_words:
        if "pulo" in pred_words:
            pulo_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD (MLM Method)")
print("="*70)
print(f"\nPolo accuracy: {polo_correct}/50 = {polo_correct/50:.2%}")
print(f"Pulo accuracy: {pulo_correct}/50 = {pulo_correct/50:.2%}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

context_accuracy = mlm_only_accuracy
mlm_only_imp = mlm_only_accuracy - baseline_accuracy
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

mlm_polo, mlm_pulo = count_word_accuracy(mlm_only_results, "polo", "pulo")

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Polo (50)  │  Pulo (50)    │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_polo:2d}/50    │    {baseline_correct_pulo:2d}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ Pure MLM-PLL               │   {mlm_only_accuracy:6.2f}%    │   {mlm_polo:2d}/50    │    {mlm_pulo:2d}/50     │
│   (MLM Scoring Only)         │ ({mlm_only_imp:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      Baseline always picks first candidate.
""")

# Save detailed results
output = {
    'ambiguous_pair': 'polo, pulo',
    'baybayin': 'ᜉᜓᜎᜓ',
    'type': 'O/U',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'polo_accuracy': f"{baseline_correct_polo}/50",
            'pulo_accuracy': f"{baseline_correct_pulo}/50"
        },
        'mlm_pll': {
            'name': 'Pure MLM-PLL (MLM Scoring Only)',
            'strategy': 'MLM pseudo-log-likelihood scoring for context-aware disambiguation',
            'accuracy': mlm_only_accuracy,
            'correct': mlm_only_metrics['correct_ambiguous'],
            'polo_accuracy': f"{mlm_polo}/50",
            'pulo_accuracy': f"{mlm_pulo}/50"
        },
        'improvement_over_baseline': mlm_only_imp
    },
    'metrics': mlm_only_metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_polo_pulo.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_polo_pulo.json")
print("="*70)
