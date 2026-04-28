"""
Test disambiguator on hito/heto ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/06_hito_heto.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with hito vs heto are mixed throughout"""
    hito_sentences = []
    heto_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "hito" in words:
            hito_sentences.append(line)
        elif "heto" in words:
            heto_sentences.append(line)
    
    return hito_sentences, heto_sentences

hito_sentences, heto_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("HITO/HETO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nHito sentences: {len(hito_sentences)}")
print(f"Heto sentences: {len(heto_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

hito_with_target = []
hito_without_target = []
heto_with_target = []
heto_without_target = []

for i, sent in enumerate(hito_sentences, 1):
    words = get_clean_words(sent)
    if "hito" in words:
        hito_with_target.append((i, sent))
    else:
        hito_without_target.append((i, sent))

for i, sent in enumerate(heto_sentences, 1):
    words = get_clean_words(sent)
    if "heto" in words:
        heto_with_target.append((i+len(hito_sentences), sent))
    else:
        heto_without_target.append((i+len(hito_sentences), sent))

print(f"\nHITO sentences with 'hito': {len(hito_with_target)}/{len(hito_sentences)}")
print(f"HETO sentences with 'heto': {len(heto_with_target)}/{len(heto_sentences)}")

if hito_without_target:
    print(f"\n⚠️  HITO sentences WITHOUT 'hito' word ({len(hito_without_target)}):")
    for line_num, sent in hito_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(hito_without_target) > 5:
        print(f"  ... and {len(hito_without_target) - 5} more")

if heto_without_target:
    print(f"\n⚠️  HETO sentences WITHOUT 'heto' word ({len(heto_without_target)}):")
    for line_num, sent in heto_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(heto_without_target) > 5:
        print(f"  ... and {len(heto_without_target) - 5} more")

# Create test data with OCR candidates
# For hito/heto, both map to Baybayin ᜑᜒᜆᜓ
# MaBaybay default order: ["hito", "heto"] (hito is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add hito sentences (ground truth = hito)
for sent in hito_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "hito":
            # Ambiguous position - both candidates (MaBaybay order: hito first)
            candidates.append(["hito", "heto"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add heto sentences (ground truth = heto)
for sent in heto_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "heto":
            # Ambiguous position - both candidates (MaBaybay order: hito first)
            candidates.append(["hito", "heto"])
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

# First candidate is always "hito" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_hito = 0
baseline_correct_heto = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "hito" (first candidate in MaBaybay order)
    if "hito" in gt_words:
        baseline_correct_total += 1
        baseline_correct_hito += 1
    # If ground truth is "heto", baseline gets it wrong (picks "hito")
    # So baseline_correct_heto stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'hito' (first candidate)")
print(f"Hito accuracy: {baseline_correct_hito}/50 = {baseline_correct_hito/50:.2%}")
print(f"Heto accuracy: {baseline_correct_heto}/50 = {baseline_correct_heto/50:.2%}")
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

print(f"\nAmbiguous words (hito/heto): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_hito_examples = []
incorrect_hito_examples = []
correct_heto_examples = []
incorrect_heto_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a hito or heto sentence
    if "hito" in gt_words:
        if "hito" in pred_words:
            correct_hito_examples.append((i+1, gt, pred))
        else:
            incorrect_hito_examples.append((i+1, gt, pred))
    elif "heto" in gt_words:
        if "heto" in pred_words:
            correct_heto_examples.append((i+1, gt, pred))
        else:
            incorrect_heto_examples.append((i+1, gt, pred))

# Display HITO results
print(f"\n{'='*70}")
print(f"HITO SENTENCES: {len(correct_hito_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_hito_examples:
    print(f"\n✓ CORRECT HITO PREDICTIONS ({len(correct_hito_examples)}):")
    for idx, gt, pred in correct_hito_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_hito_examples:
    print(f"\n✗ INCORRECT HITO PREDICTIONS ({len(incorrect_hito_examples)}):")
    for idx, gt, pred in incorrect_hito_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display HETO results
print(f"\n{'='*70}")
print(f"HETO SENTENCES: {len(correct_heto_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_heto_examples:
    print(f"\n✓ CORRECT HETO PREDICTIONS ({len(correct_heto_examples)}):")
    for idx, gt, pred in correct_heto_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_heto_examples:
    print(f"\n✗ INCORRECT HETO PREDICTIONS ({len(incorrect_heto_examples)}):")
    for idx, gt, pred in incorrect_heto_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
hito_correct = 0
heto_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "hito" in gt_words:
        if "hito" in pred_words:
            hito_correct += 1
    elif "heto" in gt_words:
        if "heto" in pred_words:
            heto_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nHito accuracy: {hito_correct}/50 = {hito_correct/50:.2%}")
print(f"Heto accuracy: {heto_correct}/50 = {heto_correct/50:.2%}")

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

mlm_hito, mlm_heto = count_word_accuracy(mlm_only_results, "hito", "heto")

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
    'ambiguous_pair': 'hito, heto',
    'baybayin': 'ᜑᜒᜆᜓ',
    'type': 'I/E',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'hito_accuracy': f"{baseline_correct_hito}/50",
            'heto_accuracy': f"{baseline_correct_heto}/50"
        },
        'cosine_only': {
            'name': 'Pure Cosine Similarity (Semantic Only)',
            'strategy': '100% cosine similarity of mean-pooled RoBERTa embeddings, no other features',
            'accuracy': cosine_only_accuracy,
            'correct': cosine_only_metrics['correct_ambiguous'],
            'hito_accuracy': f"{cosine_only_hito}/50",
            'heto_accuracy': f"{cosine_only_heto}/50"
        },
        'cosine_multi': {
            'name': 'Cosine Similarity + Multi-Feature (Old Method)',
            'strategy': 'Cosine similarity semantic + frequency + cooccurrence + morphology',
            'accuracy': cosine_multi_accuracy,
            'correct': cosine_multi_metrics['correct_ambiguous'],
            'hito_accuracy': f"{cosine_multi_hito}/50",
            'heto_accuracy': f"{cosine_multi_heto}/50"
        },
        'mlm_pll': {
            'name': 'MLM PLL + Multi-Feature (Current)',
            'strategy': 'MLM pseudo-log-likelihood semantic + frequency + cooccurrence + morphology',
            'accuracy': context_accuracy,
            'correct': metrics['correct_ambiguous'],
            'hito_accuracy': f"{mlm_hito}/50",
            'heto_accuracy': f"{mlm_heto}/50"
        },
        'improvement_over_baseline': improvement
    },
    'metrics': mlm_only_metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_hito_heto.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_hito_heto.json")
print("="*70)
