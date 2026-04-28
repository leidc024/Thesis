"""
Test disambiguator on todo/toro/turo ambiguous triplet
Compares Context-Aware Baybayin Transliteration vs MaBaybay Default (First Candidate)
Testing with 150 sentences (50 each)
NOTE: This is a 3-way ambiguity — todo, toro, and turo all map to ᜆᜓᜇᜓ
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
SENTENCE_FILE = "gold_standard_dataset/sentences/14_todo_toro_turo.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with todo vs toro vs turo"""
    todo_sentences = []
    toro_sentences = []
    turo_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "todo" in words:
            todo_sentences.append(line)
        elif "toro" in words:
            toro_sentences.append(line)
        elif "turo" in words:
            turo_sentences.append(line)
    
    return todo_sentences, toro_sentences, turo_sentences

todo_sentences, toro_sentences, turo_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("TODO/TORO/TURO DISAMBIGUATION TEST (3-WAY)")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nTodo sentences: {len(todo_sentences)}")
print(f"Toro sentences: {len(toro_sentences)}")
print(f"Turo sentences: {len(turo_sentences)}")

TOTAL_SENTENCES = len(todo_sentences) + len(toro_sentences) + len(turo_sentences)
PER_WORD = 50  # Expected per word

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

todo_with_target = []
todo_without_target = []
toro_with_target = []
toro_without_target = []
turo_with_target = []
turo_without_target = []

for i, sent in enumerate(todo_sentences, 1):
    words = get_clean_words(sent)
    if "todo" in words:
        todo_with_target.append((i, sent))
    else:
        todo_without_target.append((i, sent))

for i, sent in enumerate(toro_sentences, 1):
    words = get_clean_words(sent)
    if "toro" in words:
        toro_with_target.append((i+len(todo_sentences), sent))
    else:
        toro_without_target.append((i+len(todo_sentences), sent))

for i, sent in enumerate(turo_sentences, 1):
    words = get_clean_words(sent)
    if "turo" in words:
        turo_with_target.append((i+len(todo_sentences)+len(toro_sentences), sent))
    else:
        turo_without_target.append((i+len(todo_sentences)+len(toro_sentences), sent))

print(f"\nTODO sentences with 'todo': {len(todo_with_target)}/{len(todo_sentences)}")
print(f"TORO sentences with 'toro': {len(toro_with_target)}/{len(toro_sentences)}")
print(f"TURO sentences with 'turo': {len(turo_with_target)}/{len(turo_sentences)}")

if todo_without_target:
    print(f"\n⚠️  TODO sentences WITHOUT 'todo' word ({len(todo_without_target)}):")
    for line_num, sent in todo_without_target[:5]:
        print(f"  Line {line_num}: {sent}")
    if len(todo_without_target) > 5:
        print(f"  ... and {len(todo_without_target) - 5} more")

if toro_without_target:
    print(f"\n⚠️  TORO sentences WITHOUT 'toro' word ({len(toro_without_target)}):")
    for line_num, sent in toro_without_target[:5]:
        print(f"  Line {line_num}: {sent}")
    if len(toro_without_target) > 5:
        print(f"  ... and {len(toro_without_target) - 5} more")

if turo_without_target:
    print(f"\n⚠️  TURO sentences WITHOUT 'turo' word ({len(turo_without_target)}):")
    for line_num, sent in turo_without_target[:5]:
        print(f"  Line {line_num}: {sent}")
    if len(turo_without_target) > 5:
        print(f"  ... and {len(turo_without_target) - 5} more")

# Create test data with OCR candidates
# For todo/toro/turo, all three map to Baybayin ᜆᜓᜇᜓ
# MaBaybay default order: ["todo", "toro", "turo"] (todo is first candidate)
# NOTE: MaBaybay sends ALL words as lowercase, no punctuation (from dictionary lookup)
AMBIGUOUS_WORDS = {"todo", "toro", "turo"}
CANDIDATES = ["todo", "toro", "turo"]

test_data = []

# Add todo sentences (ground truth = todo)
for sent in todo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in AMBIGUOUS_WORDS:
            candidates.append(CANDIDATES.copy())
        else:
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add toro sentences (ground truth = toro)
for sent in toro_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in AMBIGUOUS_WORDS:
            candidates.append(CANDIDATES.copy())
        else:
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add turo sentences (ground truth = turo)
for sent in turo_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in AMBIGUOUS_WORDS:
            candidates.append(CANDIDATES.copy())
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

# First candidate is always "todo" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_todo = 0
baseline_correct_toro = 0
baseline_correct_turo = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks "todo" (first candidate in MaBaybay order)
    if "todo" in gt_words:
        baseline_correct_total += 1
        baseline_correct_todo += 1
    # If ground truth is "toro" or "turo", baseline gets it wrong (picks "todo")

baseline_accuracy = baseline_correct_total / TOTAL_SENTENCES * 100

print(f"\nBaseline Strategy: Always select 'todo' (first candidate)")
print(f"Todo accuracy: {baseline_correct_todo}/{PER_WORD} = {baseline_correct_todo/PER_WORD:.2%}")
print(f"Toro accuracy: {baseline_correct_toro}/{PER_WORD} = {baseline_correct_toro/PER_WORD:.2%}")
print(f"Turo accuracy: {baseline_correct_turo}/{PER_WORD} = {baseline_correct_turo/PER_WORD:.2%}")
print(f"Overall baseline accuracy: {baseline_correct_total}/{TOTAL_SENTENCES} = {baseline_accuracy:.2f}%")

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

print(f"\nAmbiguous words (todo/toro/turo): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_todo_examples = []
incorrect_todo_examples = []
correct_toro_examples = []
incorrect_toro_examples = []
correct_turo_examples = []
incorrect_turo_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    if "todo" in gt_words:
        if "todo" in pred_words:
            correct_todo_examples.append((i+1, gt, pred))
        else:
            incorrect_todo_examples.append((i+1, gt, pred))
    elif "toro" in gt_words:
        if "toro" in pred_words:
            correct_toro_examples.append((i+1, gt, pred))
        else:
            incorrect_toro_examples.append((i+1, gt, pred))
    elif "turo" in gt_words:
        if "turo" in pred_words:
            correct_turo_examples.append((i+1, gt, pred))
        else:
            incorrect_turo_examples.append((i+1, gt, pred))

# Display TODO results
print(f"\n{'='*70}")
print(f"TODO SENTENCES: {len(correct_todo_examples)}/{PER_WORD} CORRECT")
print(f"{'='*70}")

if correct_todo_examples:
    print(f"\n✓ CORRECT TODO PREDICTIONS ({len(correct_todo_examples)}):")
    for idx, gt, pred in correct_todo_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_todo_examples:
    print(f"\n✗ INCORRECT TODO PREDICTIONS ({len(incorrect_todo_examples)}):")
    for idx, gt, pred in incorrect_todo_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display TORO results
print(f"\n{'='*70}")
print(f"TORO SENTENCES: {len(correct_toro_examples)}/{PER_WORD} CORRECT")
print(f"{'='*70}")

if correct_toro_examples:
    print(f"\n✓ CORRECT TORO PREDICTIONS ({len(correct_toro_examples)}):")
    for idx, gt, pred in correct_toro_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_toro_examples:
    print(f"\n✗ INCORRECT TORO PREDICTIONS ({len(incorrect_toro_examples)}):")
    for idx, gt, pred in incorrect_toro_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display TURO results
print(f"\n{'='*70}")
print(f"TURO SENTENCES: {len(correct_turo_examples)}/{PER_WORD} CORRECT")
print(f"{'='*70}")

if correct_turo_examples:
    print(f"\n✓ CORRECT TURO PREDICTIONS ({len(correct_turo_examples)}):")
    for idx, gt, pred in correct_turo_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_turo_examples:
    print(f"\n✗ INCORRECT TURO PREDICTIONS ({len(incorrect_turo_examples)}):")
    for idx, gt, pred in incorrect_turo_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type for MLM method
todo_correct = 0
toro_correct = 0
turo_correct = 0
for test_item, result_item in zip(test_data, mlm_only_results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    if "todo" in gt_words:
        if "todo" in pred_words:
            todo_correct += 1
    elif "toro" in gt_words:
        if "toro" in pred_words:
            toro_correct += 1
    elif "turo" in gt_words:
        if "turo" in pred_words:
            turo_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD (MLM Method)")
print("="*70)
print(f"\nTodo accuracy: {todo_correct}/{PER_WORD} = {todo_correct/PER_WORD:.2%}")
print(f"Toro accuracy: {toro_correct}/{PER_WORD} = {toro_correct/PER_WORD:.2%}")
print(f"Turo accuracy: {turo_correct}/{PER_WORD} = {turo_correct/PER_WORD:.2%}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

context_accuracy = mlm_only_accuracy
improvement = context_accuracy - baseline_accuracy

# Count per-word accuracy for each method
def count_word_accuracy(result_list):
    w_todo = 0
    w_toro = 0
    w_turo = 0
    for test_item, result_item in zip(test_data, result_list):
        gt_words = get_clean_words(test_item['ground_truth'])
        pred_words = get_clean_words(result_item['predicted'])
        if "todo" in gt_words:
            if "todo" in pred_words:
                w_todo += 1
        elif "toro" in gt_words:
            if "toro" in pred_words:
                w_toro += 1
        elif "turo" in gt_words:
            if "turo" in pred_words:
                w_turo += 1
    return w_todo, w_toro, w_turo

cosine_only_todo, cosine_only_toro, cosine_only_turo = count_word_accuracy(cosine_only_results)
cosine_multi_todo, cosine_multi_toro, cosine_multi_turo = count_word_accuracy(cosine_multi_results)
mlm_todo, mlm_toro, mlm_turo = count_word_accuracy(mlm_results)

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
    'ambiguous_pair': 'todo, toro, turo',
    'baybayin': 'ᜆᜓᜇᜓ',
    'type': 'O/U + D/R',
    'num_candidates': 3,
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'todo_accuracy': f"{baseline_correct_todo}/{PER_WORD}",
            'toro_accuracy': f"{baseline_correct_toro}/{PER_WORD}",
            'turo_accuracy': f"{baseline_correct_turo}/{PER_WORD}"
        },
        'cosine_only': {
            'name': 'Pure Cosine Similarity (Semantic Only)',
            'strategy': '100% cosine similarity of mean-pooled RoBERTa embeddings, no other features',
            'accuracy': cosine_only_accuracy,
            'correct': cosine_only_metrics['correct_ambiguous'],
            'todo_accuracy': f"{cosine_only_todo}/{PER_WORD}",
            'toro_accuracy': f"{cosine_only_toro}/{PER_WORD}",
            'turo_accuracy': f"{cosine_only_turo}/{PER_WORD}"
        },
        'cosine_multi': {
            'name': 'Cosine Similarity + Multi-Feature (Old Method)',
            'strategy': 'Cosine similarity semantic + frequency + cooccurrence + morphology',
            'accuracy': cosine_multi_accuracy,
            'correct': cosine_multi_metrics['correct_ambiguous'],
            'todo_accuracy': f"{cosine_multi_todo}/{PER_WORD}",
            'toro_accuracy': f"{cosine_multi_toro}/{PER_WORD}",
            'turo_accuracy': f"{cosine_multi_turo}/{PER_WORD}"
        },
        'mlm_pll': {
            'name': 'MLM PLL + Multi-Feature (Current)',
            'strategy': 'MLM pseudo-log-likelihood semantic + frequency + cooccurrence + morphology',
            'accuracy': context_accuracy,
            'correct': metrics['correct_ambiguous'],
            'todo_accuracy': f"{mlm_todo}/{PER_WORD}",
            'toro_accuracy': f"{mlm_toro}/{PER_WORD}",
            'turo_accuracy': f"{mlm_turo}/{PER_WORD}"
        },
        'improvement_over_baseline': improvement
    },
    'metrics': mlm_only_metrics
}

os.makedirs("gold_standard_dataset/results", exist_ok=True)
with open("gold_standard_dataset/results/results_todo_toro_turo.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✓ Detailed results saved to: gold_standard_dataset/results/results_todo_toro_turo.json")
print("="*70)
