"""
Test disambiguator on higante/higanti ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/05_higante_higanti.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with higante vs higanti are mixed throughout"""
    higante_sentences = []
    higanti_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "higante" in words:
            higante_sentences.append(line)
        elif "higanti" in words:
            higanti_sentences.append(line)
    
    return higante_sentences, higanti_sentences

higante_sentences, higanti_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("HIGANTE/HIGANTI DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nHigante sentences: {len(higante_sentences)}")
print(f"Higanti sentences: {len(higanti_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

higante_with_target = []
higante_without_target = []
higanti_with_target = []
higanti_without_target = []

for i, sent in enumerate(higante_sentences, 1):
    words = get_clean_words(sent)
    if "higante" in words:
        higante_with_target.append((i, sent))
    else:
        higante_without_target.append((i, sent))

for i, sent in enumerate(higanti_sentences, 1):
    words = get_clean_words(sent)
    if "higanti" in words:
        higanti_with_target.append((i+len(higante_sentences), sent))
    else:
        higanti_without_target.append((i+len(higante_sentences), sent))

print(f"\nHIGANTE sentences with 'higante': {len(higante_with_target)}/{len(higante_sentences)}")
print(f"HIGANTI sentences with 'higanti': {len(higanti_with_target)}/{len(higanti_sentences)}")

if higante_without_target:
    print(f"\n⚠️  HIGANTE sentences WITHOUT 'higante' word ({len(higante_without_target)}):")
    for line_num, sent in higante_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(higante_without_target) > 5:
        print(f"  ... and {len(higante_without_target) - 5} more")

if higanti_without_target:
    print(f"\n⚠️  HIGANTI sentences WITHOUT 'higanti' word ({len(higanti_without_target)}):")
    for line_num, sent in higanti_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(higanti_without_target) > 5:
        print(f"  ... and {len(higanti_without_target) - 5} more")

# Create test data with OCR candidates
# For higante/higanti, both map to Baybayin ᜑᜒᜄᜈ᜔ᜆᜒ
# MaBaybay default order: ["higante", "higanti"] (higante is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add higante sentences (ground truth = higante)
for sent in higante_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "higante":
            # Ambiguous position - both candidates (MaBaybay order: higante first)
            candidates.append(["higante", "higanti"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add higanti sentences (ground truth = higanti)
for sent in higanti_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "higanti":
            # Ambiguous position - both candidates (MaBaybay order: higante first)
            candidates.append(["higante", "higanti"])
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

# First candidate is always "higante" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_higante = 0
baseline_correct_higanti = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "higante" (first candidate in MaBaybay order)
    if "higante" in gt_words:
        baseline_correct_total += 1
        baseline_correct_higante += 1
    # If ground truth is "higanti", baseline gets it wrong (picks "higante")
    # So baseline_correct_higanti stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'higante' (first candidate)")
print(f"Higante accuracy: {baseline_correct_higante}/50 = {baseline_correct_higante/50:.2%}")
print(f"Higanti accuracy: {baseline_correct_higanti}/50 = {baseline_correct_higanti/50:.2%}")
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

print(f"\nAmbiguous words (higante/higanti): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_higante_examples = []
incorrect_higante_examples = []
correct_higanti_examples = []
incorrect_higanti_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a higante or higanti sentence
    if "higante" in gt_words:
        if "higante" in pred_words:
            correct_higante_examples.append((i+1, gt, pred))
        else:
            incorrect_higante_examples.append((i+1, gt, pred))
    elif "higanti" in gt_words:
        if "higanti" in pred_words:
            correct_higanti_examples.append((i+1, gt, pred))
        else:
            incorrect_higanti_examples.append((i+1, gt, pred))

# Display HIGANTE results
print(f"\n{'='*70}")
print(f"HIGANTE SENTENCES: {len(correct_higante_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_higante_examples:
    print(f"\n✓ CORRECT HIGANTE PREDICTIONS ({len(correct_higante_examples)}):")
    for idx, gt, pred in correct_higante_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_higante_examples:
    print(f"\n✗ INCORRECT HIGANTE PREDICTIONS ({len(incorrect_higante_examples)}):")
    for idx, gt, pred in incorrect_higante_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display HIGANTI results
print(f"\n{'='*70}")
print(f"HIGANTI SENTENCES: {len(correct_higanti_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_higanti_examples:
    print(f"\n✓ CORRECT HIGANTI PREDICTIONS ({len(correct_higanti_examples)}):")
    for idx, gt, pred in correct_higanti_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_higanti_examples:
    print(f"\n✗ INCORRECT HIGANTI PREDICTIONS ({len(incorrect_higanti_examples)}):")
    for idx, gt, pred in incorrect_higanti_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
higante_correct = 0
higanti_correct = 0
for test_item, result_item in zip(test_data, mlm_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "higante" in gt_words:
        if "higante" in pred_words:
            higante_correct += 1
    elif "higanti" in gt_words:
        if "higanti" in pred_words:
            higanti_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD (MLM Method)")
print("="*70)
print(f"\nHigante accuracy: {higante_correct}/50 = {higante_correct/50:.2%}")
print(f"Higanti accuracy: {higanti_correct}/50 = {higanti_correct/50:.2%}")

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

cosine_only_higante, cosine_only_higanti = count_word_accuracy(cosine_only_results, "higante", "higanti")
cosine_multi_higante, cosine_multi_higanti = count_word_accuracy(cosine_multi_results, "higante", "higanti")
mlm_higante, mlm_higanti = count_word_accuracy(mlm_results, "higante", "higanti")

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │Higante (50)│ Higanti (50)  │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_higante:2d}/50    │    {baseline_correct_higanti:2d}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ Pure Cosine Similarity       │   {cosine_only_accuracy:6.2f}%    │   {cosine_only_higante:2d}/50    │    {cosine_only_higanti:2d}/50     │
│ (Semantic Only)              │ ({cosine_only_imp:+6.2f}%)  │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ Cosine Sim + Multi-Feature   │   {cosine_multi_accuracy:6.2f}%    │   {cosine_multi_higante:2d}/50    │    {cosine_multi_higanti:2d}/50     │
│ (Old Method)                 │ ({cosine_multi_imp:+6.2f}%)  │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ MLM PLL + Multi-Feature    │   {context_accuracy:6.2f}%    │   {mlm_higante:2d}/50    │    {mlm_higanti:2d}/50     │
│   (Current Method)           │ ({improvement:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      For 'ᜑᜒᜄᜈ᜔ᜆᜒ', candidates are ["higante", "higanti"], so baseline always picks "higante".
""")

# Save detailed results
output = {
    'ambiguous_pair': 'higante, higanti',
    'baybayin': 'ᜑᜒᜄᜈ᜔ᜆᜒ',
    'type': 'E/I',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'higante_accuracy': f"{baseline_correct_higante}/50",
            'higanti_accuracy': f"{baseline_correct_higanti}/50"
        },
        'cosine_only': {
            'name': 'Pure Cosine Similarity (Semantic Only)',
            'strategy': '100% cosine similarity of mean-pooled RoBERTa embeddings, no other features',
            'accuracy': cosine_only_accuracy,
            'correct': cosine_only_metrics['correct_ambiguous'],
            'cosine_only_higante': f"{cosine_only_higante}/50",
            'cosine_only_higanti': f"{cosine_only_higanti}/50"
        },
        'cosine_multi': {
            'name': 'Cosine Similarity + Multi-Feature (Old Method)',
            'strategy': 'Cosine similarity semantic + frequency + cooccurrence + morphology',
            'accuracy': cosine_multi_accuracy,
            'correct': cosine_multi_metrics['correct_ambiguous'],
            'higante_accuracy': f"{cosine_multi_higante}/50",
            'higanti_accuracy': f"{cosine_multi_higanti}/50"
        },
        'mlm_multi': {
            'name': 'MLM PLL + Multi-Feature (Current)',
            'strategy': 'MLM pseudo-log-likelihood semantic + frequency + cooccurrence + morphology',
            'accuracy': context_accuracy,
            'correct': metrics['correct_ambiguous'],
            'higante_accuracy': f"{mlm_higante}/50",
            'higanti_accuracy': f"{mlm_higanti}/50"
        },
        'improvement_over_baseline': improvement
    },
    'metrics': metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_higante_higanti.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_higante_higanti.json")
print("="*70)
