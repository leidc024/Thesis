"""
Test disambiguator on mesa/misa ambiguous pair
Compares Context-Aware Baybayin Transliteration vs MaBaybay Default (First Candidate)
Testing with 100 sentences (50 each)
"""

import json
import re
import sys
import os
import time
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
SENTENCE_FILE = "gold_standard_dataset/sentences/10_mesa_misa.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with mesa vs misa are mixed throughout"""
    mesa_sentences = []
    misa_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "mesa" in words:
            mesa_sentences.append(line)
        elif "misa" in words:
            misa_sentences.append(line)
    
    return mesa_sentences, misa_sentences

mesa_sentences, misa_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("MESA/MISA DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nMesa sentences: {len(mesa_sentences)}")
print(f"Misa sentences: {len(misa_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

mesa_with_target = []
mesa_without_target = []
misa_with_target = []
misa_without_target = []

for i, sent in enumerate(mesa_sentences, 1):
    words = get_clean_words(sent)
    if "mesa" in words:
        mesa_with_target.append((i, sent))
    else:
        mesa_without_target.append((i, sent))

for i, sent in enumerate(misa_sentences, 1):
    words = get_clean_words(sent)
    if "misa" in words:
        misa_with_target.append((i+len(mesa_sentences), sent))
    else:
        misa_without_target.append((i+len(mesa_sentences), sent))

print(f"\nMESA sentences with 'mesa': {len(mesa_with_target)}/{len(mesa_sentences)}")
print(f"MISA sentences with 'misa': {len(misa_with_target)}/{len(misa_sentences)}")

if mesa_without_target:
    print(f"\n⚠️  MESA sentences WITHOUT 'mesa' word ({len(mesa_without_target)}):")
    for line_num, sent in mesa_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(mesa_without_target) > 5:
        print(f"  ... and {len(mesa_without_target) - 5} more")

if misa_without_target:
    print(f"\n⚠️  MISA sentences WITHOUT 'misa' word ({len(misa_without_target)}):")
    for line_num, sent in misa_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(misa_without_target) > 5:
        print(f"  ... and {len(misa_without_target) - 5} more")

# Create test data with OCR candidates
# For mesa/misa, both map to Baybayin ᜋᜒᜐ
# MaBaybay default order: ["mesa", "misa"] (mesa is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add mesa sentences (ground truth = mesa)
for sent in mesa_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "mesa":
            # Ambiguous position - both candidates (MaBaybay order: mesa first)
            candidates.append(["mesa", "misa"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add misa sentences (ground truth = misa)
for sent in misa_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "misa":
            # Ambiguous position - both candidates (MaBaybay order: mesa first)
            candidates.append(["mesa", "misa"])
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

# First candidate is always "mesa" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_mesa = 0
baseline_correct_misa = 0

baseline_start_time = time.time()
for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "mesa" (first candidate in MaBaybay order)
    if "mesa" in gt_words:
        baseline_correct_total += 1
        baseline_correct_mesa += 1
    # If ground truth is "misa", baseline gets it wrong (picks "mesa")
    # So baseline_correct_misa stays 0

baseline_time = time.time() - baseline_start_time
baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'mesa' (first candidate)")
print(f"Mesa accuracy: {baseline_correct_mesa}/50 = {baseline_correct_mesa/50:.2%}")
print(f"Misa accuracy: {baseline_correct_misa}/50 = {baseline_correct_misa/50:.2%}")
print(f"Overall baseline accuracy: {baseline_correct_total}/100 = {baseline_accuracy:.2f}%")

# ============================================================================
# INITIALIZE MODEL (shared across all methods)
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

all_test_sentences = [item['ground_truth'] for item in test_data]
init_start_time = time.time()
model = BaybayinDisambiguator(
    corpus_files=[
        "corpus/Tagalog_Literary_Text.txt",
        "corpus/Tagalog_Religious_Text.txt",
        "corpus/Tagalog_Balita_Texts_Balanced.txt"
    ],
    exclude_sentences=all_test_sentences  # Clean evaluation - no data leakage
)
init_time = time.time() - init_start_time
print(f"Initialization time: {init_time:.2f}s")

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

eval_start_time = time.time()
mlm_only_metrics, mlm_only_results = model.evaluate(
    test_data, show_progress=True, use_mlm=True, weights_override=mlm_only_weights
)
eval_time = time.time() - eval_start_time
mlm_only_accuracy = mlm_only_metrics['ambiguous_accuracy'] * 100
time_per_ambiguity = eval_time / mlm_only_metrics['total_ambiguous']
print(f"★ Pure MLM-PLL accuracy: {mlm_only_accuracy:.2f}%")
print(f"Evaluation time: {eval_time:.2f}s ({time_per_ambiguity*1000:.2f}ms per ambiguity)")

# Use MLM-PLL results for detailed predictions display
metrics = mlm_only_metrics
results = mlm_only_results

# Display results
print("\n" + "="*70)
print("CONTEXT-AWARE DISAMBIGUATION RESULTS")
print("="*70)

print(f"\nAmbiguous words (mesa/misa): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_mesa_examples = []
incorrect_mesa_examples = []
correct_misa_examples = []
incorrect_misa_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a mesa or misa sentence
    if "mesa" in gt_words:
        if "mesa" in pred_words:
            correct_mesa_examples.append((i+1, gt, pred))
        else:
            incorrect_mesa_examples.append((i+1, gt, pred))
    elif "misa" in gt_words:
        if "misa" in pred_words:
            correct_misa_examples.append((i+1, gt, pred))
        else:
            incorrect_misa_examples.append((i+1, gt, pred))

# Display MESA results
print(f"\n{'='*70}")
print(f"MESA SENTENCES: {len(correct_mesa_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_mesa_examples:
    print(f"\n✓ CORRECT MESA PREDICTIONS ({len(correct_mesa_examples)}):")
    for idx, gt, pred in correct_mesa_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_mesa_examples:
    print(f"\n✗ INCORRECT MESA PREDICTIONS ({len(incorrect_mesa_examples)}):")
    for idx, gt, pred in incorrect_mesa_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display MISA results
print(f"\n{'='*70}")
print(f"MISA SENTENCES: {len(correct_misa_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_misa_examples:
    print(f"\n✓ CORRECT MISA PREDICTIONS ({len(correct_misa_examples)}):")
    for idx, gt, pred in correct_misa_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_misa_examples:
    print(f"\n✗ INCORRECT MISA PREDICTIONS ({len(incorrect_misa_examples)}):")
    for idx, gt, pred in incorrect_misa_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method (used in detailed display above)
mesa_correct = 0
misa_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "mesa" in gt_words:
        if "mesa" in pred_words:
            mesa_correct += 1
    elif "misa" in gt_words:
        if "misa" in pred_words:
            misa_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nMesa accuracy: {mesa_correct}/50 = {mesa_correct/50:.2%}")
print(f"Misa accuracy: {misa_correct}/50 = {misa_correct/50:.2%}")

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

mlm_mesa, mlm_misa = count_word_accuracy(mlm_only_results, "mesa", "misa")

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Mesa (50)  │  Misa (50)    │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_mesa:2d}/50    │    {baseline_correct_misa:2d}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ Pure MLM-PLL               │   {mlm_only_accuracy:6.2f}%    │   {mlm_mesa:2d}/50    │    {mlm_misa:2d}/50     │
│   (MLM Scoring Only)         │ ({mlm_only_imp:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      Baseline always picks first candidate.
""")

# Save detailed results
output = {
    'ambiguous_pair': 'mesa, misa',
    'baybayin': 'ᜋᜒᜐ',
    'type': 'E/I',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'mesa_accuracy': f"{baseline_correct_mesa}/50",
            'misa_accuracy': f"{baseline_correct_misa}/50"
        },
        'mlm_pll': {
            'name': 'Pure MLM-PLL (MLM Scoring Only)',
            'strategy': 'MLM pseudo-log-likelihood scoring for context-aware disambiguation',
            'accuracy': mlm_only_accuracy,
            'correct': mlm_only_metrics['correct_ambiguous'],
            'mesa_accuracy': f"{mlm_mesa}/50",
            'misa_accuracy': f"{mlm_misa}/50"
        },
        'improvement_over_baseline': mlm_only_imp
    },
    'runtime_metrics': {
        'baseline': {
            'method': 'MaBaybay Default (First Candidate)',
            'execution_time_seconds': baseline_time,
            'milliseconds_per_ambiguous_position': (baseline_time / 100) * 1000
        },
        'mlm_pll': {
            'method': 'Pure MLM-PLL',
            'initialization_time_seconds': init_time,
            'execution_time_seconds': eval_time,
            'total_time_seconds': init_time + eval_time,
            'milliseconds_per_ambiguous_position': time_per_ambiguity * 1000,
            'ambiguous_positions_evaluated': mlm_only_metrics['total_ambiguous']
        }
    },
    'metrics': mlm_only_metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_mesa_misa.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_mesa_misa.json")
print("="*70)
