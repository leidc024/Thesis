"""
Test disambiguator on kumita/kometa ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/09_kumita_kometa.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with kumita vs kometa are mixed throughout"""
    kumita_sentences = []
    kometa_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "kumita" in words:
            kumita_sentences.append(line)
        elif "kometa" in words:
            kometa_sentences.append(line)
    
    return kumita_sentences, kometa_sentences

kumita_sentences, kometa_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("KUMITA/KOMETA DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nKumita sentences: {len(kumita_sentences)}")
print(f"Kometa sentences: {len(kometa_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

kumita_with_target = []
kumita_without_target = []
kometa_with_target = []
kometa_without_target = []

for i, sent in enumerate(kumita_sentences, 1):
    words = get_clean_words(sent)
    if "kumita" in words:
        kumita_with_target.append((i, sent))
    else:
        kumita_without_target.append((i, sent))

for i, sent in enumerate(kometa_sentences, 1):
    words = get_clean_words(sent)
    if "kometa" in words:
        kometa_with_target.append((i+len(kumita_sentences), sent))
    else:
        kometa_without_target.append((i+len(kumita_sentences), sent))

print(f"\nKUMITA sentences with 'kumita': {len(kumita_with_target)}/{len(kumita_sentences)}")
print(f"KOMETA sentences with 'kometa': {len(kometa_with_target)}/{len(kometa_sentences)}")

if kumita_without_target:
    print(f"\n⚠️  KUMITA sentences WITHOUT 'kumita' word ({len(kumita_without_target)}):")
    for line_num, sent in kumita_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(kumita_without_target) > 5:
        print(f"  ... and {len(kumita_without_target) - 5} more")

if kometa_without_target:
    print(f"\n⚠️  KOMETA sentences WITHOUT 'kometa' word ({len(kometa_without_target)}):")
    for line_num, sent in kometa_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(kometa_without_target) > 5:
        print(f"  ... and {len(kometa_without_target) - 5} more")

# Create test data with OCR candidates
# For kumita/kometa, both map to Baybayin ᜃᜓᜋᜒᜆ
# MaBaybay default order: ["kumita", "kometa"] (kumita is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add kumita sentences (ground truth = kumita)
for sent in kumita_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "kumita":
            # Ambiguous position - both candidates (MaBaybay order: kumita first)
            candidates.append(["kumita", "kometa"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add kometa sentences (ground truth = kometa)
for sent in kometa_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "kometa":
            # Ambiguous position - both candidates (MaBaybay order: kumita first)
            candidates.append(["kumita", "kometa"])
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

# First candidate is always "kumita" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_kumita = 0
baseline_correct_kometa = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "kumita" (first candidate in MaBaybay order)
    if "kumita" in gt_words:
        baseline_correct_total += 1
        baseline_correct_kumita += 1
    # If ground truth is "kometa", baseline gets it wrong (picks "kumita")
    # So baseline_correct_kometa stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'kumita' (first candidate)")
print(f"Kumita accuracy: {baseline_correct_kumita}/50 = {baseline_correct_kumita/50:.2%}")
print(f"Kometa accuracy: {baseline_correct_kometa}/50 = {baseline_correct_kometa/50:.2%}")
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

print(f"\nAmbiguous words (kumita/kometa): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_kumita_examples = []
incorrect_kumita_examples = []
correct_kometa_examples = []
incorrect_kometa_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a kumita or kometa sentence
    if "kumita" in gt_words:
        if "kumita" in pred_words:
            correct_kumita_examples.append((i+1, gt, pred))
        else:
            incorrect_kumita_examples.append((i+1, gt, pred))
    elif "kometa" in gt_words:
        if "kometa" in pred_words:
            correct_kometa_examples.append((i+1, gt, pred))
        else:
            incorrect_kometa_examples.append((i+1, gt, pred))

# Display KUMITA results
print(f"\n{'='*70}")
print(f"KUMITA SENTENCES: {len(correct_kumita_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_kumita_examples:
    print(f"\n✓ CORRECT KUMITA PREDICTIONS ({len(correct_kumita_examples)}):")
    for idx, gt, pred in correct_kumita_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_kumita_examples:
    print(f"\n✗ INCORRECT KUMITA PREDICTIONS ({len(incorrect_kumita_examples)}):")
    for idx, gt, pred in incorrect_kumita_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display KOMETA results
print(f"\n{'='*70}")
print(f"KOMETA SENTENCES: {len(correct_kometa_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_kometa_examples:
    print(f"\n✓ CORRECT KOMETA PREDICTIONS ({len(correct_kometa_examples)}):")
    for idx, gt, pred in correct_kometa_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_kometa_examples:
    print(f"\n✗ INCORRECT KOMETA PREDICTIONS ({len(incorrect_kometa_examples)}):")
    for idx, gt, pred in incorrect_kometa_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
kumita_correct = 0
kometa_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "kumita" in gt_words:
        if "kumita" in pred_words:
            kumita_correct += 1
    elif "kometa" in gt_words:
        if "kometa" in pred_words:
            kometa_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nKumita accuracy: {kumita_correct}/50 = {kumita_correct/50:.2%}")
print(f"Kometa accuracy: {kometa_correct}/50 = {kometa_correct/50:.2%}")

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

mlm_kumita, mlm_kometa = count_word_accuracy(mlm_only_results, "kumita", "kometa")

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Kumita (50) │  Kometa (50)  │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_kumita:2d}/50    │    {baseline_correct_kometa:2d}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ Pure MLM-PLL               │   {mlm_only_accuracy:6.2f}%    │   {mlm_kumita:2d}/50    │    {mlm_kometa:2d}/50     │
│   (MLM Scoring Only)         │ ({mlm_only_imp:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      Baseline always picks first candidate.
""")

# Save detailed results
output = {
    'ambiguous_pair': 'kumita, kometa',
    'baybayin': 'ᜃᜓᜋᜒᜆ',
    'type': 'U/O',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'kumita_accuracy': f"{baseline_correct_kumita}/50",
            'kometa_accuracy': f"{baseline_correct_kometa}/50"
        },
        'mlm_pll': {
            'name': 'Pure MLM-PLL (MLM Scoring Only)',
            'strategy': 'MLM pseudo-log-likelihood scoring for context-aware disambiguation',
            'accuracy': mlm_only_accuracy,
            'correct': mlm_only_metrics['correct_ambiguous'],
            'kumita_accuracy': f"{mlm_kumita}/50",
            'kometa_accuracy': f"{mlm_kometa}/50"
        },
        'improvement_over_baseline': mlm_only_imp
    },
    'metrics': mlm_only_metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_kumita_kometa.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_kumita_kometa.json")
print("="*70)
