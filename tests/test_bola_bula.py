"""
Test disambiguator on bola/bula ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/02_bola_bula.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with bola vs bula are mixed throughout"""
    bola_sentences = []
    bula_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "bola" in words:
            bola_sentences.append(line)
        elif "bula" in words:
            bula_sentences.append(line)
    
    return bola_sentences, bula_sentences

bola_sentences, bula_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("BOLA/BULA DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nBola sentences: {len(bola_sentences)}")
print(f"Bula sentences: {len(bula_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

bola_with_target = []
bola_without_target = []
bula_with_target = []
bula_without_target = []

for i, sent in enumerate(bola_sentences, 1):
    words = get_clean_words(sent)
    if "bola" in words:
        bola_with_target.append((i, sent))
    else:
        bola_without_target.append((i, sent))

for i, sent in enumerate(bula_sentences, 1):
    words = get_clean_words(sent)
    if "bula" in words:
        bula_with_target.append((i+len(bola_sentences), sent))
    else:
        bula_without_target.append((i+len(bola_sentences), sent))

print(f"\nBOLA sentences with 'bola': {len(bola_with_target)}/{len(bola_sentences)}")
print(f"BULA sentences with 'bula': {len(bula_with_target)}/{len(bula_sentences)}")

if bola_without_target:
    print(f"\n⚠️  BOLA sentences WITHOUT 'bola' word ({len(bola_without_target)}):")
    for line_num, sent in bola_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(bola_without_target) > 5:
        print(f"  ... and {len(bola_without_target) - 5} more")

if bula_without_target:
    print(f"\n⚠️  BULA sentences WITHOUT 'bula' word ({len(bula_without_target)}):")
    for line_num, sent in bula_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(bula_without_target) > 5:
        print(f"  ... and {len(bula_without_target) - 5} more")

# Create test data with OCR candidates
# For bola/bula, both map to Baybayin ᜊᜓᜎ
# MaBaybay default order: ["bola", "bula"] (bola is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add bola sentences (ground truth = bola)
for sent in bola_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "bola":
            # Ambiguous position - both candidates (MaBaybay order: bola first)
            candidates.append(["bola", "bula"])
        else:
            # Unambiguous word (lowercase, no punctuation - same as MaBaybay)
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add bula sentences (ground truth = bula)
for sent in bula_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Simulate MaBaybay output: lowercase, no punctuation (dictionary words)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "bula":
            # Ambiguous position - both candidates (MaBaybay order: bola first)
            candidates.append(["bola", "bula"])
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

# First candidate is always "bola" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_bola = 0
baseline_correct_bula = 0

baseline_start_time = time.time()
for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Check if sentence contains target words
    if "bola" in gt_words:
        baseline_correct_total += 1
        baseline_correct_bola += 1
    # If ground truth is "bula", baseline gets it wrong (picks "bola")
    # So baseline_correct_bula stays 0

baseline_time = time.time() - baseline_start_time
baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'bola' (first candidate)")
print(f"Bola accuracy: {baseline_correct_bola}/50 = {baseline_correct_bola/50:.2%}")
print(f"Bula accuracy: {baseline_correct_bula}/50 = {baseline_correct_bula/50:.2%}")
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
print("\n" + "="*70)
print("METHOD 1: PURE MLM-PLL (Semantic Only)")
print("="*70)

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

print(f"\nAmbiguous words (bola/bula): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_bola_examples = []
incorrect_bola_examples = []
correct_bula_examples = []
incorrect_bula_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a bola or bula sentence
    if "bola" in gt_words:
        if "bola" in pred_words:
            correct_bola_examples.append((i+1, gt, pred))
        else:
            incorrect_bola_examples.append((i+1, gt, pred))
    elif "bula" in gt_words:
        if "bula" in pred_words:
            correct_bula_examples.append((i+1, gt, pred))
        else:
            incorrect_bula_examples.append((i+1, gt, pred))

# Display BOLA results
print(f"\n{'='*70}")
print(f"BOLA SENTENCES: {len(correct_bola_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_bola_examples:
    print(f"\n✓ CORRECT BOLA PREDICTIONS ({len(correct_bola_examples)}):")
    for idx, gt, pred in correct_bola_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_bola_examples:
    print(f"\n✗ INCORRECT BOLA PREDICTIONS ({len(incorrect_bola_examples)}):")
    for idx, gt, pred in incorrect_bola_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display BULA results
print(f"\n{'='*70}")
print(f"BULA SENTENCES: {len(correct_bula_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_bula_examples:
    print(f"\n✓ CORRECT BULA PREDICTIONS ({len(correct_bula_examples)}):")
    for idx, gt, pred in correct_bula_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_bula_examples:
    print(f"\n✗ INCORRECT BULA PREDICTIONS ({len(incorrect_bula_examples)}):")
    for idx, gt, pred in incorrect_bula_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type - computed in comparison section below
bola_correct = 0
bula_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "bola" in gt_words:
        if "bola" in pred_words:
            bola_correct += 1
    elif "bula" in gt_words:
        if "bula" in pred_words:
            bula_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nBola accuracy: {bola_correct}/50 = {bola_correct/50:.2%}")
print(f"Bula accuracy: {bula_correct}/50 = {bula_correct/50:.2%}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

context_accuracy = mlm_only_accuracy
improvement = context_accuracy - baseline_accuracy

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

mlm_only_bola, mlm_only_bula = count_word_accuracy(mlm_only_results, "bola", "bula")

mlm_only_imp = mlm_only_accuracy - baseline_accuracy

print(f"""
┌──────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Bola (50)  │  Bula (50)    │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_bola:2d}/50    │    {baseline_correct_bula:2d}/50     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ Pure MLM-PLL               │   {mlm_only_accuracy:6.2f}%    │   {mlm_only_bola:2d}/50    │    {mlm_only_bula:2d}/50     │
│   (MLM Scoring Only)         │ ({mlm_only_imp:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      For 'ᜊᜓᜎ', candidates are ["bola", "bula"], so baseline always picks "bola".
""")

# Save detailed results
output = {
    'ambiguous_pair': 'bola, bula',
    'baybayin': 'ᜊᜓᜎ',
    'type': 'O/U',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'bola_accuracy': f"{baseline_correct_bola}/50",
            'bula_accuracy': f"{baseline_correct_bula}/50"
        },
        'mlm_pll': {
            'name': 'Pure MLM-PLL (MLM Scoring Only)',
            'strategy': 'MLM pseudo-log-likelihood scoring for context-aware disambiguation',
            'accuracy': mlm_only_accuracy,
            'correct': mlm_only_metrics['correct_ambiguous'],
            'bola_accuracy': f"{mlm_only_bola}/50",
            'bula_accuracy': f"{mlm_only_bula}/50"
        },
        'improvement_over_baseline': improvement
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
with open("gold_standard_dataset/results/results_bola_bula.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_bola_bula.json")
print("="*70)
