"""
Test disambiguator on bote/buti ambiguous pair
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
SENTENCE_FILE = "gold_standard_dataset/sentences/02_bote_buti.txt"

def parse_sentence_file(filepath):
    """Parse sentence file - sentences with bote vs buti are mixed throughout"""
    bote_sentences = []
    buti_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Separate based on exact word match only (case-insensitive, ignore punctuation)
    for line in lines:
        words = get_clean_words(line)
        if "bote" in words:
            bote_sentences.append(line)
        elif "buti" in words:
            buti_sentences.append(line)
    
    return bote_sentences, buti_sentences

bote_sentences, buti_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("BOTE/BUTI DISAMBIGUATION TEST")
print("Comparing: Context-Aware Disambiguation vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nBote sentences: {len(bote_sentences)}")
print(f"Buti sentences: {len(buti_sentences)}")

# Debug: Check which sentences contain target words
print("\n" + "="*50)
print("DEBUGGING: Checking for target words")
print("="*50)

bote_with_target = []
bote_without_target = []
buti_with_target = []
buti_without_target = []

for i, sent in enumerate(bote_sentences, 1):
    words = get_clean_words(sent)
    if "bote" in words:
        bote_with_target.append((i, sent))
    else:
        bote_without_target.append((i, sent))

for i, sent in enumerate(buti_sentences, 1):
    words = get_clean_words(sent)
    if "buti" in words:
        buti_with_target.append((i+len(bote_sentences), sent))
    else:
        buti_without_target.append((i+len(bote_sentences), sent))

print(f"\nBOTE sentences with 'bote': {len(bote_with_target)}/{len(bote_sentences)}")
print(f"BUTI sentences with 'buti': {len(buti_with_target)}/{len(buti_sentences)}")

if bote_without_target:
    print(f"\n⚠️  BOTE sentences WITHOUT 'bote' word ({len(bote_without_target)}):")
    for line_num, sent in bote_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(bote_without_target) > 5:
        print(f"  ... and {len(bote_without_target) - 5} more")

if buti_without_target:
    print(f"\n⚠️  BUTI sentences WITHOUT 'buti' word ({len(buti_without_target)}):")
    for line_num, sent in buti_without_target[:5]:  # Show first 5
        print(f"  Line {line_num}: {sent}")
    if len(buti_without_target) > 5:
        print(f"  ... and {len(buti_without_target) - 5} more")

# Create test data with OCR candidates
# For bote/buti, both map to Baybayin ᜊᜓᜆᜒ
# MaBaybay default order: ["bote", "buti"] (bote is first candidate)
test_data = []

# Add bote sentences (ground truth = bote)
for sent in bote_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Strip punctuation for comparison
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word == "bote":
            # Ambiguous position - both candidates
            candidates.append(["bote", "buti"])
        else:
            # Unambiguous word
            candidates.append(word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

# Add buti sentences (ground truth = buti)
for sent in buti_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        # Strip punctuation for comparison
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word == "buti":
            # Ambiguous position - both candidates
            candidates.append(["bote", "buti"])
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

# First candidate is always "bote" in MaBaybay's transliteration output
baseline_correct_total = 0
baseline_correct_bote = 0
baseline_correct_buti = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Check if sentence contains target words
    if "bote" in gt_words:
        baseline_correct_total += 1
        baseline_correct_bote += 1
    # If ground truth is "buti", baseline gets it wrong (picks "bote")
    # So baseline_correct_buti stays 0

baseline_accuracy = baseline_correct_total / 100 * 100  # 100 total sentences

print(f"\nBaseline Strategy: Always select 'bote' (first candidate)")
print(f"Bote accuracy: {baseline_correct_bote}/50 = {baseline_correct_bote/50:.2%}")
print(f"Buti accuracy: {baseline_correct_buti}/50 = {baseline_correct_buti/50:.2%}")
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

print(f"\nAmbiguous words (bote/buti): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("DETAILED PREDICTIONS - ALL RESULTS")
print("="*70)

# Collect ALL examples, categorized
correct_bote_examples = []
incorrect_bote_examples = []
correct_buti_examples = []
incorrect_buti_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    gt_words = get_clean_words(gt)
    pred_words = get_clean_words(pred)
    
    # Check if this is a bote or buti sentence
    if "bote" in gt_words:
        if "bote" in pred_words:
            correct_bote_examples.append((i+1, gt, pred))
        else:
            incorrect_bote_examples.append((i+1, gt, pred))
    elif "buti" in gt_words:
        if "buti" in pred_words:
            correct_buti_examples.append((i+1, gt, pred))
        else:
            incorrect_buti_examples.append((i+1, gt, pred))

# Display BOTE results
print(f"\n{'='*70}")
print(f"BOTE SENTENCES: {len(correct_bote_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_bote_examples:
    print(f"\n✓ CORRECT BOTE PREDICTIONS ({len(correct_bote_examples)}):")
    for idx, gt, pred in correct_bote_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_bote_examples:
    print(f"\n✗ INCORRECT BOTE PREDICTIONS ({len(incorrect_bote_examples)}):")
    for idx, gt, pred in incorrect_bote_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Display BUTI results
print(f"\n{'='*70}")
print(f"BUTI SENTENCES: {len(correct_buti_examples)}/50 CORRECT")
print(f"{'='*70}")

if correct_buti_examples:
    print(f"\n✓ CORRECT BUTI PREDICTIONS ({len(correct_buti_examples)}):")
    for idx, gt, pred in correct_buti_examples:
        print(f"\n{idx}. ✓ {gt}")

if incorrect_buti_examples:
    print(f"\n✗ INCORRECT BUTI PREDICTIONS ({len(incorrect_buti_examples)}):")
    for idx, gt, pred in incorrect_buti_examples:
        print(f"\n{idx}. ✗ Ground Truth: {gt}")
        print(f"      Predicted:    {pred}")

# Breakdown by word type (exact word match only, ignoring punctuation)
bote_correct = 0
buti_correct = 0

for test_item, result_item in zip(test_data, results):
    gt_words = get_clean_words(test_item['ground_truth'])
    pred_words = get_clean_words(result_item['predicted'])
    
    # Check if this is a bote sentence (exact word)
    if "bote" in gt_words:
        if "bote" in pred_words:
            bote_correct += 1
    # Check if this is a buti sentence (exact word)
    elif "buti" in gt_words:
        if "buti" in pred_words:
            buti_correct += 1

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nBote accuracy: {bote_correct}/50 = {bote_correct/50:.2%}")
print(f"Buti accuracy: {buti_correct}/50 = {buti_correct/50:.2%}")

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
│ MaBaybay Default        │    {baseline_accuracy:6.2f}%       │ Always picks 'bote'   │
│ (First Candidate)       │                  │ (first candidate)     │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ Context-Aware           │    {graph_accuracy:6.2f}%       │ Uses RoBERTa-Tagalog  │
│ Disambiguation          │                  │ and context features  │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ ★ Improvement           │   +{improvement:6.2f}%       │                       │
└─────────────────────────┴──────────────────┴───────────────────────┘

Breakdown by Word:
  • Bote sentences: Disambiguation={bote_correct}/50 vs Baseline={baseline_correct_bote}/50
  • Buti sentences: Disambiguation={buti_correct}/50 vs Baseline={baseline_correct_buti}/50

Note: MaBaybay default always returns first candidate from transliteration.
      For 'ᜊᜓᜆᜒ', candidates are ["bote", "buti"], so baseline always picks "bote".
""")

# Save detailed results
output = {
    'ambiguous_pair': 'bote, buti',
    'baybayin': 'ᜊᜓᜆᜒ',
    'type': 'COMBINED (O/U + E/I)',
    'test_sentences': len(test_data),
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'strategy': 'Always pick first candidate from transliteration',
            'accuracy': baseline_accuracy,
            'correct': baseline_correct_total,
            'bote_accuracy': f"{baseline_correct_bote}/50",
            'buti_accuracy': f"{baseline_correct_buti}/50"
        },
        'context_aware': {
            'name': 'Context-Aware Baybayin Transliteration',
            'strategy': 'Multi-feature approach using RoBERTa-Tagalog, frequency, co-occurrence, morphology',
            'accuracy': graph_accuracy,
            'correct': metrics['correct_ambiguous'],
            'bote_accuracy': f"{bote_correct}/50",
            'buti_accuracy': f"{buti_correct}/50"
        },
        'improvement': improvement
    },
    'metrics': metrics
}

with open("results_bote_buti.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Detailed results saved to: results_bote_buti.json")
print("="*70)
