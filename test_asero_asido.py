"""
Test disambiguator on asero/asido ambiguous pair
Compares Context-Aware Baybayin Transliteration vs MaBaybay Default (First Candidate)
Testing with 100 sentences (50 each)
"""

import json
import re
from src.disambiguator import BaybayinDisambiguator

def get_clean_words(sentence):
    """Extract words from sentence, removing punctuation"""
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', sentence.lower())
    return words

# Read the sentences from gold standard dataset
SENTENCE_FILE = "gold_standard_dataset/sentences/01_asero_asido.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with asero vs asido are mixed throughout"""
    asero_sentences = []
    asido_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "asero" in words:
            asero_sentences.append(line)
        elif "asido" in words:
            asido_sentences.append(line)
    
    return asero_sentences, asido_sentences

asero_sentences, asido_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("ASERO/ASIDO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nAsero sentences: {len(asero_sentences)}")
print(f"Asido sentences: {len(asido_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

asero_with_target = []
asero_without_target = []
asido_with_target = []
asido_without_target = []

for i, sent in enumerate(asero_sentences, 1):
    words = get_clean_words(sent)
    if "asero" in words:
        asero_with_target.append((i, sent))
    else:
        asero_without_target.append((i, sent))

for i, sent in enumerate(asido_sentences, 1):
    words = get_clean_words(sent)
    if "asido" in words:
        asido_with_target.append((i+len(asero_sentences), sent))
    else:
        asido_without_target.append((i+len(asero_sentences), sent))

print(f"\nASERO sentences with 'asero': {len(asero_with_target)}/{len(asero_sentences)}")
print(f"ASIDO sentences with 'asido': {len(asido_with_target)}/{len(asido_sentences)}")

if asero_without_target:
    print(f"\n⚠️  ASERO sentences WITHOUT 'asero' word ({len(asero_without_target)}):")
    for line_num, sent in asero_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(asero_without_target) > 5:
        print(f"  ... and {len(asero_without_target) - 5} more")

if asido_without_target:
    print(f"\n⚠️  ASIDO sentences WITHOUT 'asido' word ({len(asido_without_target)}):")
    for line_num, sent in asido_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(asido_without_target) > 5:
        print(f"  ... and {len(asido_without_target) - 5} more")

# Create test data with OCR candidates
# For asero/asido, both map to Baybayin ᜀᜐᜒᜇᜓ
# MaBaybay default order: ["asero", "asido"] (asero is first candidate)
test_data = []

# Add asero sentences (ground truth = asero)
for sent in asero_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        if word.lower() == "asero":
            # Ambiguous position - both candidates
            candidates.append(["asero", "asido"])
        else:
            # Unambiguous word
            candidates.append(word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add asido sentences (ground truth = asido)
for sent in asido_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        if word.lower() == "asido":
            # Ambiguous position - both candidates
            candidates.append(["asero", "asido"])
        else:
            # Unambiguous word
            candidates.append(word)
    
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

# First candidate is always "asero" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_asero = 0
baseline_correct_asido = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Check if sentence contains target words
    if "asero" in gt_words:
        baseline_correct_total += 1
        baseline_correct_asero += 1
    # If ground truth is "asido", baseline gets it wrong (picks "asero")
    # So baseline_correct_asido stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'asero' (first candidate)")
print(f"Asero accuracy: {baseline_correct_asero}/50 = {baseline_correct_asero/50:.2%}")
print(f"Asido accuracy: {baseline_correct_asido}/50 = {baseline_correct_asido/50:.2%}")
print(f"Overall baseline accuracy: {baseline_correct_total}/100 = {baseline_accuracy:.2f}%")

# ============================================================================
# CONTEXT-AWARE DISAMBIGUATION
# ============================================================================
print("\n" + "="*70)
print("CONTEXT-AWARE DISAMBIGUATION MODEL")
print("="*70)

all_test_sentences = [item['ground_truth'] for item in test_data]
model = BaybayinDisambiguator(
    corpus_files=[
        "Tagalog_Literary_Text.txt",
        "Tagalog_Religious_Text.txt",
        "Tagalog_Balita_Texts_Balanced.txt"
    ],
    exclude_sentences=all_test_sentences  # Clean evaluation - no data leakage
)

# Run evaluation
print("\nRunning disambiguation on test sentences...")

metrics, results = model.evaluate(test_data, show_progress=True)

# Display results
print("\n" + "="*70)
print("CONTEXT-AWARE DISAMBIGUATION RESULTS")
print("="*70)

print(f"\nAmbiguous words (asero/asido): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_asero_examples = []
incorrect_asero_examples = []
correct_asido_examples = []
incorrect_asido_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a asero or asido sentence
    if "asero" in gt_words:
        if "asero" in pred_words:
            correct_asero_examples.append((i+1, gt, pred))
        else:
            incorrect_asero_examples.append((i+1, gt, pred))
    elif "asido" in gt_words:
        if "asido" in pred_words:
            correct_asido_examples.append((i+1, gt, pred))
        else:
            incorrect_asido_examples.append((i+1, gt, pred))

# Display ASERO results
print(f"\n{'='*70}")
print(f"ASERO SENTENCES: {len(correct_asero_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_asero_examples:
    print(f"\n✓ CORRECT ASERO PREDICTIONS ({len(correct_asero_examples)}):")
    for idx, gt, pred in correct_asero_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_asero_examples:
    print(f"\n✗ INCORRECT ASERO PREDICTIONS ({len(incorrect_asero_examples)}):")
    for idx, gt, pred in incorrect_asero_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display ASIDO results
print(f"\n{'='*70}")
print(f"ASIDO SENTENCES: {len(correct_asido_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_asido_examples:
    print(f"\n✓ CORRECT ASIDO PREDICTIONS ({len(correct_asido_examples)}):")
    for idx, gt, pred in correct_asido_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_asido_examples:
    print(f"\n✗ INCORRECT ASIDO PREDICTIONS ({len(incorrect_asido_examples)}):")
    for idx, gt, pred in incorrect_asido_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type (exact word match only, ignoring punctuation)
asero_correct = 0
asido_correct = 0

for test_item, result_item in zip(test_data, results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    
    # Check if this is a asero sentence (exact word)
    if "asero" in gt_words:
        if "asero" in pred_words:
            asero_correct += 1
    # Check if this is a asido sentence (exact word)
    elif "asido" in gt_words:
        if "asido" in pred_words:
            asido_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nAsero accuracy: {asero_correct}/50 = {asero_correct/50:.2%}")
print(f"Asido accuracy: {asido_correct}/50 = {asido_correct/50:.2%}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

graph_accuracy = metrics['ambiguous_accuracy'] * 100
improvement = graph_accuracy - baseline_accuracy

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                    DISAMBIGUATION RESULTS                          │
├─────────────────────────┬──────────────────┬───────────────────────┤
│        Method           │    Accuracy      │       Details         │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ MaBaybay Default        │    {baseline_accuracy:6.2f}%       │ Always picks 'asero'  │
│ (First Candidate)       │                  │ (first candidate)     │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ Context-Aware           │    {graph_accuracy:6.2f}%       │ Uses RoBERTa-Tagalog  │
│ Disambiguation          │                  │ and context features  │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ ★ Improvement           │   +{improvement:6.2f}%       │                       │
└─────────────────────────┴──────────────────┴───────────────────────┘

Breakdown by Word:
  • Asero sentences: Disambiguation={asero_correct}/50 vs Baseline={baseline_correct_asero}/50
  • Asido sentences: Disambiguation={asido_correct}/50 vs Baseline={baseline_correct_asido}/50

Note: MaBaybay default always returns first candidate from transliteration.
      For 'ᜀᜐᜒᜇᜓ', candidates are ["asero", "asido"], so baseline always picks "asero".
""")

# Save detailed results
output = {
    'ambiguous_pair': 'asero, asido',
    'baybayin': 'ᜀᜐᜒᜇᜓ',
    'type': 'E/I + O/U',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'asero_accuracy': f"{baseline_correct_asero}/50",
            'asido_accuracy': f"{baseline_correct_asido}/50"
        },
        'context_aware': {
            'name': 'Context-Aware Baybayin Transliteration',
            'strategy': 'Multi-feature approach using RoBERTa-Tagalog, frequency, co-occurrence, morphology',
            'accuracy': graph_accuracy,
            'correct': metrics['correct_ambiguous'],
            'asero_accuracy': f"{asero_correct}/50",
            'asido_accuracy': f"{asido_correct}/50"
        },
        'improvement': improvement
    },
    'metrics': metrics
}

with open("results_asero_asido.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Detailed results saved to: results_asero_asido.json")
print("="*70)
