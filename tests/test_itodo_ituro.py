"""
Test disambiguator on itodo/ituro ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/07_itodo_ituro.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with itodo vs ituro are mixed throughout"""
    itodo_sentences = []
    ituro_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "itodo" in words:
            itodo_sentences.append(line)
        elif "ituro" in words:
            ituro_sentences.append(line)
    
    return itodo_sentences, ituro_sentences

itodo_sentences, ituro_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("ITODO/ITURO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nItodo sentences: {len(itodo_sentences)}")
print(f"Ituro sentences: {len(ituro_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

itodo_with_target = []
itodo_without_target = []
ituro_with_target = []
ituro_without_target = []

for i, sent in enumerate(itodo_sentences, 1):
    words = get_clean_words(sent)
    if "itodo" in words:
        itodo_with_target.append((i, sent))
    else:
        itodo_without_target.append((i, sent))

for i, sent in enumerate(ituro_sentences, 1):
    words = get_clean_words(sent)
    if "ituro" in words:
        ituro_with_target.append((i+len(itodo_sentences), sent))
    else:
        ituro_without_target.append((i+len(itodo_sentences), sent))

print(f"\nITODO sentences with 'itodo': {len(itodo_with_target)}/{len(itodo_sentences)}")
print(f"ITURO sentences with 'ituro': {len(ituro_with_target)}/{len(ituro_sentences)}")

if itodo_without_target:
    print(f"\n⚠️  ITODO sentences WITHOUT 'itodo' word ({len(itodo_without_target)}):")
    for line_num, sent in itodo_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(itodo_without_target) > 5:
        print(f"  ... and {len(itodo_without_target) - 5} more")

if ituro_without_target:
    print(f"\n⚠️  ITURO sentences WITHOUT 'ituro' word ({len(ituro_without_target)}):")
    for line_num, sent in ituro_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(ituro_without_target) > 5:
        print(f"  ... and {len(ituro_without_target) - 5} more")

# Create test data with OCR candidates
# For itodo/ituro, both map to Baybayin ᜁᜆᜓᜇᜓ
# MaBaybay default order: ["itodo", "ituro"] (itodo is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add itodo sentences (ground truth = itodo)
for sent in itodo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "itodo":
            # Ambiguous position - both candidates (MaBaybay order: itodo first)
            candidates.append(["itodo", "ituro"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add ituro sentences (ground truth = ituro)
for sent in ituro_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "ituro":
            # Ambiguous position - both candidates (MaBaybay order: itodo first)
            candidates.append(["itodo", "ituro"])
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

# First candidate is always "itodo" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_itodo = 0
baseline_correct_ituro = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "itodo" (first candidate in MaBaybay order)
    if "itodo" in gt_words:
        baseline_correct_total += 1
        baseline_correct_itodo += 1
    # If ground truth is "ituro", baseline gets it wrong (picks "itodo")
    # So baseline_correct_ituro stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'itodo' (first candidate)")
print(f"Itodo accuracy: {baseline_correct_itodo}/50 = {baseline_correct_itodo/50:.2%}")
print(f"Ituro accuracy: {baseline_correct_ituro}/50 = {baseline_correct_ituro/50:.2%}")
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
# METHOD 1: Pure Cosine Similarity (Semantic Only, No Other Features)
# ============================================================================
print("\n" + "="*70)
print("METHOD 1: PURE COSINE SIMILARITY (Semantic Only)")
print("="*70)

cosine_only_weights = {
    'semantic': 1.0,
    'frequency': 0.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}
print(f"\nWeights: {cosine_only_weights}")
print("Semantic scoring: Cosine similarity of mean-pooled RoBERTa embeddings")
print("Running evaluation...")

cosine_only_metrics, cosine_only_results = model.evaluate(
    test_data, show_progress=True, use_mlm=False, weights_override=cosine_only_weights
)
cosine_only_accuracy = cosine_only_metrics['ambiguous_accuracy'] * 100
print(f"★ Pure Cosine Similarity accuracy: {cosine_only_accuracy:.2f}%")

# ============================================================================
# METHOD 2: Cosine Similarity + Multi-Feature (Old Method)
# ============================================================================
print("\n" + "="*70)
print("METHOD 2: COSINE SIMILARITY + MULTI-FEATURE (Old Method)")
print("="*70)

print(f"\nWeights: semantic=0.4, frequency=0.3, cooccurrence=0.2, morphology=0.1")
print("Semantic scoring: Cosine similarity of mean-pooled RoBERTa embeddings")
print("Running evaluation...")

cosine_multi_metrics, cosine_multi_results = model.evaluate(
    test_data, show_progress=True, use_mlm=False
)
cosine_multi_accuracy = cosine_multi_metrics['ambiguous_accuracy'] * 100
print(f"★ Cosine Multi-Feature accuracy: {cosine_multi_accuracy:.2f}%")

# ============================================================================
# METHOD 3: MLM Pseudo-Log-Likelihood + Multi-Feature (Current Best)
# ============================================================================
print("\n" + "="*70)
print("METHOD 3: MLM PSEUDO-LOG-LIKELIHOOD + MULTI-FEATURE (Current)")
print("="*70)

print(f"\nWeights: semantic=0.4, frequency=0.3, cooccurrence=0.2, morphology=0.1")
print("Semantic scoring: MLM PLL (Masked Language Model Pseudo-Log-Likelihood)")
print("Running evaluation...")

mlm_metrics, mlm_results = model.evaluate(
    test_data, show_progress=True, use_mlm=True
)
mlm_accuracy = mlm_metrics['ambiguous_accuracy'] * 100
print(f"★ MLM Multi-Feature accuracy: {mlm_accuracy:.2f}%")

# Use MLM results (best method) for detailed predictions display
metrics = mlm_metrics
results = mlm_results

# Display results
print("\n" + "="*70)
print("CONTEXT-AWARE DISAMBIGUATION RESULTS")
print("="*70)

print(f"\nAmbiguous words (itodo/ituro): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_itodo_examples = []
incorrect_itodo_examples = []
correct_ituro_examples = []
incorrect_ituro_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is an itodo or ituro sentence
    if "itodo" in gt_words:
        if "itodo" in pred_words:
            correct_itodo_examples.append((i+1, gt, pred))
        else:
            incorrect_itodo_examples.append((i+1, gt, pred))
    elif "ituro" in gt_words:
        if "ituro" in pred_words:
            correct_ituro_examples.append((i+1, gt, pred))
        else:
            incorrect_ituro_examples.append((i+1, gt, pred))

# Display ITODO results
print(f"\n{'='*70}")
print(f"ITODO SENTENCES: {len(correct_itodo_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_itodo_examples:
    print(f"\n✓ CORRECT ITODO PREDICTIONS ({len(correct_itodo_examples)}):")
    for idx, gt, pred in correct_itodo_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_itodo_examples:
    print(f"\n✗ INCORRECT ITODO PREDICTIONS ({len(incorrect_itodo_examples)}):")
    for idx, gt, pred in incorrect_itodo_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display ITURO results
print(f"\n{'='*70}")
print(f"ITURO SENTENCES: {len(correct_ituro_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_ituro_examples:
    print(f"\n✓ CORRECT ITURO PREDICTIONS ({len(correct_ituro_examples)}):")
    for idx, gt, pred in correct_ituro_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_ituro_examples:
    print(f"\n✗ INCORRECT ITURO PREDICTIONS ({len(incorrect_ituro_examples)}):")
    for idx, gt, pred in incorrect_ituro_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
itodo_correct = 0
ituro_correct = 0
for test_item, result_item in zip(test_data, mlm_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "itodo" in gt_words:
        if "itodo" in pred_words:
            itodo_correct += 1
    elif "ituro" in gt_words:
        if "ituro" in pred_words:
            ituro_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD (MLM Method)")
print("="*70)
print(f"\nItodo accuracy: {itodo_correct}/50 = {itodo_correct/50:.2%}")
print(f"Ituro accuracy: {ituro_correct}/50 = {ituro_correct/50:.2%}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

context_accuracy = mlm_accuracy
improvement = context_accuracy - baseline_accuracy
cosine_only_imp = cosine_only_accuracy - baseline_accuracy
cosine_multi_imp = cosine_multi_accuracy - baseline_accuracy

# Count per-word accuracy for each method
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

cosine_only_itodo, cosine_only_ituro = count_word_accuracy(cosine_only_results, "itodo", "ituro")
cosine_multi_itodo, cosine_multi_ituro = count_word_accuracy(cosine_multi_results, "itodo", "ituro")
mlm_itodo, mlm_ituro = count_word_accuracy(mlm_results, "itodo", "ituro")

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Itodo (50) │  Ituro (50)   │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_itodo:2d}/50    │    {baseline_correct_ituro:2d}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ Pure Cosine Similarity       │   {cosine_only_accuracy:6.2f}%    │   {cosine_only_itodo:2d}/50    │    {cosine_only_ituro:2d}/50     │
│ (Semantic Only)              │ ({cosine_only_imp:+6.2f}%)  │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ Cosine Sim + Multi-Feature   │   {cosine_multi_accuracy:6.2f}%    │   {cosine_multi_itodo:2d}/50    │    {cosine_multi_ituro:2d}/50     │
│ (Old Method)                 │ ({cosine_multi_imp:+6.2f}%)  │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ MLM PLL + Multi-Feature    │   {context_accuracy:6.2f}%    │   {mlm_itodo:2d}/50    │    {mlm_ituro:2d}/50     │
│   (Current Method)           │ ({improvement:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      For 'ᜁᜆᜓᜇᜓ', candidates are ["itodo", "ituro"], so baseline always picks "itodo".
""")

# Save detailed results
output = {
    'ambiguous_pair': 'itodo, ituro',
    'baybayin': 'ᜁᜆᜓᜇᜓ',
    'type': 'D/R',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'itodo_accuracy': f"{baseline_correct_itodo}/50",
            'ituro_accuracy': f"{baseline_correct_ituro}/50"
        },
        'cosine_only': {
            'name': 'Pure Cosine Similarity (Semantic Only)',
            'strategy': '100% cosine similarity of mean-pooled RoBERTa embeddings, no other features',
            'accuracy': cosine_only_accuracy,
            'correct': cosine_only_metrics['correct_ambiguous'],
            'itodo_accuracy': f"{cosine_only_itodo}/50",
            'ituro_accuracy': f"{cosine_only_ituro}/50"
        },
        'cosine_multi': {
            'name': 'Cosine Similarity + Multi-Feature (Old Method)',
            'strategy': 'Cosine similarity semantic + frequency + cooccurrence + morphology',
            'accuracy': cosine_multi_accuracy,
            'correct': cosine_multi_metrics['correct_ambiguous'],
            'itodo_accuracy': f"{cosine_multi_itodo}/50",
            'ituro_accuracy': f"{cosine_multi_ituro}/50"
        },
        'mlm_multi': {
            'name': 'MLM PLL + Multi-Feature (Current)',
            'strategy': 'MLM pseudo-log-likelihood semantic + frequency + cooccurrence + morphology',
            'accuracy': context_accuracy,
            'correct': metrics['correct_ambiguous'],
            'itodo_accuracy': f"{mlm_itodo}/50",
            'ituro_accuracy': f"{mlm_ituro}/50"
        },
        'improvement_over_baseline': improvement
    },
    'metrics': metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_itodo_ituro.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_itodo_ituro.json")
print("="*70)
