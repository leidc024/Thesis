"""
Test disambiguator on toyo/tuyo ambiguous pair
Compares Context-Aware Baybayin Transliteration vs MaBaybay Default (First Candidate)
Testing with 150 sentences:
  - 50 toyo (soy sauce)
  - 50 tuyo (dried fish)
  - 50 tuyo (dry/adjective)
NOTE: tuyo has two distinct senses but both are the same word "tuyo" for disambiguation.
      From Baybayin perspective, this is a 2-candidate ambiguity: toyo vs tuyo.
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
SENTENCE_FILE = "gold_standard_dataset/sentences/15_toyo_tuyo.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with toyo vs tuyo (two senses)"""
    toyo_sentences = []
    tuyo_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "toyo" in words:
            toyo_sentences.append(line)
        elif "tuyo" in words:
            tuyo_sentences.append(line)
    
    return toyo_sentences, tuyo_sentences

toyo_sentences, tuyo_sentences = parse_sentence_file(SENTENCE_FILE)

TOTAL_SENTENCES = len(toyo_sentences) + len(tuyo_sentences)
TOYO_COUNT = len(toyo_sentences)
TUYO_COUNT = len(tuyo_sentences)

print(f"="*70)
print("TOYO/TUYO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nToyo sentences (soy sauce): {TOYO_COUNT}")
print(f"Tuyo sentences (dried fish + dry): {TUYO_COUNT}")
print(f"  Note: tuyo has 2 senses — dried fish (50) and dry/adjective (50)")
print(f"  Both senses are the same word 'tuyo' for disambiguation purposes.")
print(f"Total sentences: {TOTAL_SENTENCES}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

toyo_with_target = []
toyo_without_target = []
tuyo_with_target = []
tuyo_without_target = []

for i, sent in enumerate(toyo_sentences, 1):
    words = get_clean_words(sent)
    if "toyo" in words:
        toyo_with_target.append((i, sent))
    else:
        toyo_without_target.append((i, sent))

for i, sent in enumerate(tuyo_sentences, 1):
    words = get_clean_words(sent)
    if "tuyo" in words:
        tuyo_with_target.append((i+TOYO_COUNT, sent))
    else:
        tuyo_without_target.append((i+TOYO_COUNT, sent))

print(f"\nTOYO sentences with 'toyo': {len(toyo_with_target)}/{TOYO_COUNT}")
print(f"TUYO sentences with 'tuyo': {len(tuyo_with_target)}/{TUYO_COUNT}")

if toyo_without_target:
    print(f"\n⚠️  TOYO sentences WITHOUT 'toyo' word ({len(toyo_without_target)}):")
    for line_num, sent in toyo_without_target[:5]:
        print(f"  Line {line_num}: {sent}")
    if len(toyo_without_target) > 5:
        print(f"  ... and {len(toyo_without_target) - 5} more")

if tuyo_without_target:
    print(f"\n⚠️  TUYO sentences WITHOUT 'tuyo' word ({len(tuyo_without_target)}):")
    for line_num, sent in tuyo_without_target[:5]:
        print(f"  Line {line_num}: {sent}")
    if len(tuyo_without_target) > 5:
        print(f"  ... and {len(tuyo_without_target) - 5} more")

# Create test data with OCR candidates
# For toyo/tuyo, both map to Baybayin ᜆᜓᜌᜓ
# MaBaybay default order: ["toyo", "tuyo"] (toyo is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
test_data = []

# Add toyo sentences (ground truth = toyo)
for sent in toyo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "toyo":
            candidates.append(["toyo", "tuyo"])
        else:
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add tuyo sentences (ground truth = tuyo — both dried fish and dry senses)
for sent in tuyo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word == "tuyo":
            candidates.append(["toyo", "tuyo"])
        else:
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

# First candidate is always "toyo" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_toyo = 0
baseline_correct_tuyo = 0

baseline_start_time = time.time()
for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "toyo" (first candidate in MaBaybay order)
    if "toyo" in gt_words:
        baseline_correct_total += 1
        baseline_correct_toyo += 1
    # If ground truth is "tuyo", baseline gets it wrong (picks "toyo")

baseline_time = time.time() - baseline_start_time
baseline_accuracy = baseline_correct_total / TOTAL_SENTENCES * 100

print(f"\nBaseline Strategy: Always select 'toyo' (first candidate)")
print(f"Toyo accuracy: {baseline_correct_toyo}/{TOYO_COUNT} = {baseline_correct_toyo/TOYO_COUNT:.2%}")
print(f"Tuyo accuracy: {baseline_correct_tuyo}/{TUYO_COUNT} = {baseline_correct_tuyo/TUYO_COUNT:.2%}")
print(f"Overall baseline accuracy: {baseline_correct_total}/{TOTAL_SENTENCES} = {baseline_accuracy:.2f}%")

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

print(f"\nAmbiguous words (toyo/tuyo): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_toyo_examples = []
incorrect_toyo_examples = []
correct_tuyo_examples = []
incorrect_tuyo_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    if "toyo" in gt_words:
        if "toyo" in pred_words:
            correct_toyo_examples.append((i+1, gt, pred))
        else:
            incorrect_toyo_examples.append((i+1, gt, pred))
    elif "tuyo" in gt_words:
        if "tuyo" in pred_words:
            correct_tuyo_examples.append((i+1, gt, pred))
        else:
            incorrect_tuyo_examples.append((i+1, gt, pred))

# Display TOYO results
print(f"\n{'='*70}")
print(f"TOYO (soy sauce) SENTENCES: {len(correct_toyo_examples)}/{TOYO_COUNT} CORRECT")
print(f"{'='*70}")

if correct_toyo_examples:
    print(f"\n✓ CORRECT TOYO PREDICTIONS ({len(correct_toyo_examples)}):")
    for idx, gt, pred in correct_toyo_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_toyo_examples:
    print(f"\n✗ INCORRECT TOYO PREDICTIONS ({len(incorrect_toyo_examples)}):")
    for idx, gt, pred in incorrect_toyo_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display TUYO results — split into dried fish (first 50) and dry (next 50) for clarity
print(f"\n{'='*70}")
print(f"TUYO (dried fish + dry) SENTENCES: {len(correct_tuyo_examples)}/{TUYO_COUNT} CORRECT")
print(f"{'='*70}")

# Separate tuyo results into dried fish vs dry senses for detailed display
# Tuyo dried fish = sentences 51-100 (indices TOYO_COUNT to TOYO_COUNT+49)
# Tuyo dry = sentences 101-150 (indices TOYO_COUNT+50 to end)
tuyo_fish_correct = []
tuyo_fish_incorrect = []
tuyo_dry_correct = []
tuyo_dry_incorrect = []

for idx, gt, pred in correct_tuyo_examples:
    if idx <= TOYO_COUNT + 50:  # First 50 tuyo sentences = dried fish
        tuyo_fish_correct.append((idx, gt, pred))
    else:
        tuyo_dry_correct.append((idx, gt, pred))

for idx, gt, pred in incorrect_tuyo_examples:
    if idx <= TOYO_COUNT + 50:
        tuyo_fish_incorrect.append((idx, gt, pred))
    else:
        tuyo_dry_incorrect.append((idx, gt, pred))

print(f"\n--- TUYO (dried fish) ---")
print(f"Correct: {len(tuyo_fish_correct)}/50")

if tuyo_fish_correct:
    print(f"\n✓ CORRECT TUYO (dried fish) PREDICTIONS ({len(tuyo_fish_correct)}):")
    for idx, gt, pred in tuyo_fish_correct:
        print(f"\n{idx}. ✓ {gt}")

if tuyo_fish_incorrect:
    print(f"\n✗ INCORRECT TUYO (dried fish) PREDICTIONS ({len(tuyo_fish_incorrect)}):")
    for idx, gt, pred in tuyo_fish_incorrect:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

print(f"\n--- TUYO (dry) ---")
print(f"Correct: {len(tuyo_dry_correct)}/50")

if tuyo_dry_correct:
    print(f"\n✓ CORRECT TUYO (dry) PREDICTIONS ({len(tuyo_dry_correct)}):")
    for idx, gt, pred in tuyo_dry_correct:
        print(f"\n{idx}. ✓ {gt}")

if tuyo_dry_incorrect:
    print(f"\n✗ INCORRECT TUYO (dry) PREDICTIONS ({len(tuyo_dry_incorrect)}):")
    for idx, gt, pred in tuyo_dry_incorrect:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method
toyo_correct = 0
tuyo_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "toyo" in gt_words:
        if "toyo" in pred_words:
            toyo_correct += 1
    elif "tuyo" in gt_words:
        if "tuyo" in pred_words:
            tuyo_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD (MLM Method)")
print("="*70)
print(f"\nToyo accuracy (soy sauce):     {toyo_correct}/{TOYO_COUNT} = {toyo_correct/TOYO_COUNT:.2%}")
print(f"Tuyo accuracy (both senses):   {tuyo_correct}/{TUYO_COUNT} = {tuyo_correct/TUYO_COUNT:.2%}")
print(f"  - Tuyo (dried fish):         {len(tuyo_fish_correct)}/50")
print(f"  - Tuyo (dry):                {len(tuyo_dry_correct)}/50")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

context_accuracy = mlm_only_accuracy
mlm_only_imp = mlm_only_accuracy - baseline_accuracy
improvement = context_accuracy - baseline_accuracy

# Count per-word accuracy for each method
def count_word_accuracy(result_list):
    w_toyo = 0
    w_tuyo = 0
    for test_item, result_item in zip(test_data, result_list):
        gt_words = get_clean_words(test_item['ground_truth'])
        pred_words = get_clean_words(result_item['predicted'])
        if "toyo" in gt_words:
            if "toyo" in pred_words:
                w_toyo += 1
        elif "tuyo" in gt_words:
            if "tuyo" in pred_words:
                w_tuyo += 1
    return w_toyo, w_tuyo

mlm_toyo, mlm_tuyo = count_word_accuracy(mlm_only_results)

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Toyo (50)  │  Tuyo (50)    │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │   {baseline_accuracy:6.2f}%    │   {baseline_correct_toyo:2d}/{TOYO_COUNT}    │    {baseline_correct_tuyo:2d}/{TUYO_COUNT}     │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ ★ Pure MLM-PLL               │   {mlm_only_accuracy:6.2f}%    │   {mlm_toyo:2d}/{TOYO_COUNT}    │    {mlm_tuyo:2d}/{TUYO_COUNT}     │
│   (MLM Scoring Only)         │ ({mlm_only_imp:+6.2f}%)  │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘

Note: MaBaybay default always returns first candidate from transliteration.
      Baseline always picks first candidate.
""")

# Save detailed results
output = {
    'ambiguous_pair': 'toyo, tuyo',
    'baybayin': 'ᜆᜓᜌᜓ',
    'type': 'O/U',
    'note': 'tuyo has 2 senses: dried fish (50 sentences) and dry/adjective (50 sentences)',
    'test_sentences': TOTAL_SENTENCES,
    'toyo_count': TOYO_COUNT,
    'tuyo_count': TUYO_COUNT,
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'toyo_accuracy': f"{baseline_correct_toyo}/{TOYO_COUNT}",
            'tuyo_accuracy': f"{baseline_correct_tuyo}/{TUYO_COUNT}"
        },
        'mlm_pll': {
            'name': 'Pure MLM-PLL (MLM Scoring Only)',
            'strategy': 'MLM pseudo-log-likelihood scoring for context-aware disambiguation',
            'accuracy': mlm_only_accuracy,
            'correct': mlm_only_metrics['correct_ambiguous'],
            'toyo_accuracy': f"{mlm_toyo}/{TOYO_COUNT}",
            'tuyo_accuracy': f"{mlm_tuyo}/{TUYO_COUNT}",
            'tuyo_fish_correct': len(tuyo_fish_correct),
            'tuyo_dry_correct': len(tuyo_dry_correct)
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
with open("gold_standard_dataset/results/results_toyo_tuyo.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_toyo_tuyo.json")
print("="*70)
