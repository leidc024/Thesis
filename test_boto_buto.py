"""
Test disambiguator on boto/buto ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/03_boto_buto.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with boto vs buto are mixed throughout"""
    boto_sentences = []
    buto_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "boto" in words:
            boto_sentences.append(line)
        elif "buto" in words:
            buto_sentences.append(line)
    
    return boto_sentences, buto_sentences

boto_sentences, buto_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("BOTO/BUTO DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nBoto sentences: {len(boto_sentences)}")
print(f"Buto sentences: {len(buto_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

boto_with_target = []
boto_without_target = []
buto_with_target = []
buto_without_target = []

for i, sent in enumerate(boto_sentences, 1):
    words = get_clean_words(sent)
    if "boto" in words:
        boto_with_target.append((i, sent))
    else:
        boto_without_target.append((i, sent))

for i, sent in enumerate(buto_sentences, 1):
    words = get_clean_words(sent)
    if "buto" in words:
        buto_with_target.append((i+len(boto_sentences), sent))
    else:
        buto_without_target.append((i+len(boto_sentences), sent))

print(f"\nBOTO sentences with 'boto': {len(boto_with_target)}/{len(boto_sentences)}")
print(f"BUTO sentences with 'buto': {len(buto_with_target)}/{len(buto_sentences)}")

if boto_without_target:
    print(f"\n⚠️  BOTO sentences WITHOUT 'boto' word ({len(boto_without_target)}):")
    for line_num, sent in boto_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(boto_without_target) > 5:
        print(f"  ... and {len(boto_without_target) - 5} more")

if buto_without_target:
    print(f"\n⚠️  BUTO sentences WITHOUT 'buto' word ({len(buto_without_target)}):")
    for line_num, sent in buto_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(buto_without_target) > 5:
        print(f"  ... and {len(buto_without_target) - 5} more")

# Create test data with OCR candidates
# For boto/buto, both map to Baybayin ᜊᜓᜆᜓ
# MaBaybay default order: ["boto", "buto"] (boto is first candidate)
test_data = []

# Add boto sentences (ground truth = boto)
for sent in boto_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        if word.lower() == "boto":
            # Ambiguous position - both candidates
            candidates.append(["boto", "buto"])
        else:
            # Unambiguous word
            candidates.append(word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add buto sentences (ground truth = buto)
for sent in buto_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        if word.lower() == "buto":
            # Ambiguous position - both candidates
            candidates.append(["boto", "buto"])
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

# First candidate is always "boto" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_boto = 0
baseline_correct_buto = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Check if sentence contains target words
    if "boto" in gt_words:
        baseline_correct_total += 1
        baseline_correct_boto += 1
    # If ground truth is "buto", baseline gets it wrong (picks "boto")
    # So baseline_correct_buto stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'boto' (first candidate)")
print(f"Boto accuracy: {baseline_correct_boto}/50 = {baseline_correct_boto/50:.2%}")
print(f"Buto accuracy: {baseline_correct_buto}/50 = {baseline_correct_buto/50:.2%}")
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

print(f"\nAmbiguous words (boto/buto): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_boto_examples = []
incorrect_boto_examples = []
correct_buto_examples = []
incorrect_buto_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a boto or buto sentence
    if "boto" in gt_words:
        if "boto" in pred_words:
            correct_boto_examples.append((i+1, gt, pred))
        else:
            incorrect_boto_examples.append((i+1, gt, pred))
    elif "buto" in gt_words:
        if "buto" in pred_words:
            correct_buto_examples.append((i+1, gt, pred))
        else:
            incorrect_buto_examples.append((i+1, gt, pred))

# Display BOTO results
print(f"\n{'='*70}")
print(f"BOTO SENTENCES: {len(correct_boto_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_boto_examples:
    print(f"\n✓ CORRECT BOTO PREDICTIONS ({len(correct_boto_examples)}):")
    for idx, gt, pred in correct_boto_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_boto_examples:
    print(f"\n✗ INCORRECT BOTO PREDICTIONS ({len(incorrect_boto_examples)}):")
    for idx, gt, pred in incorrect_boto_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display BUTO results
print(f"\n{'='*70}")
print(f"BUTO SENTENCES: {len(correct_buto_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_buto_examples:
    print(f"\n✓ CORRECT BUTO PREDICTIONS ({len(correct_buto_examples)}):")
    for idx, gt, pred in correct_buto_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_buto_examples:
    print(f"\n✗ INCORRECT BUTO PREDICTIONS ({len(incorrect_buto_examples)}):")
    for idx, gt, pred in incorrect_buto_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type (exact word match only, ignoring punctuation)
boto_correct = 0
buto_correct = 0

for test_item, result_item in zip(test_data, results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    
    # Check if this is a boto sentence (exact word)
    if "boto" in gt_words:
        if "boto" in pred_words:
            boto_correct += 1
    # Check if this is a buto sentence (exact word)
    elif "buto" in gt_words:
        if "buto" in pred_words:
            buto_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nBoto accuracy: {boto_correct}/50 = {boto_correct/50:.2%}")
print(f"Buto accuracy: {buto_correct}/50 = {buto_correct/50:.2%}")

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
│ MaBaybay Default        │    {baseline_accuracy:6.2f}%       │ Always picks 'boto'   │
│ (First Candidate)       │                  │ (first candidate)     │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ Context-Aware           │    {graph_accuracy:6.2f}%       │ Uses RoBERTa-Tagalog  │
│ Disambiguation          │                  │ and context features  │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ ★ Improvement           │   +{improvement:6.2f}%       │                       │
└─────────────────────────┴──────────────────┴───────────────────────┘

Breakdown by Word:
  • Boto sentences: Disambiguation={boto_correct}/50 vs Baseline={baseline_correct_boto}/50
  • Buto sentences: Disambiguation={buto_correct}/50 vs Baseline={baseline_correct_buto}/50

Note: MaBaybay default always returns first candidate from transliteration.
      For 'ᜊᜓᜆᜓ', candidates are ["boto", "buto"], so baseline always picks "boto".
""")

# Save detailed results
output = {
    'ambiguous_pair': 'boto, buto',
    'baybayin': 'ᜊᜓᜆᜓ',
    'type': 'O/U',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'boto_accuracy': f"{baseline_correct_boto}/50",
            'buto_accuracy': f"{baseline_correct_buto}/50"
        },
        'context_aware': {
            'name': 'Context-Aware Baybayin Transliteration',
            'strategy': 'Multi-feature approach using RoBERTa-Tagalog, frequency, co-occurrence, morphology',
            'accuracy': graph_accuracy,
            'correct': metrics['correct_ambiguous'],
            'boto_accuracy': f"{boto_correct}/50",
            'buto_accuracy': f"{buto_correct}/50"
        },
        'improvement': improvement
    },
    'metrics': metrics
}

with open("results_boto_buto.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Detailed results saved to: results_boto_buto.json")
print("="*70)
