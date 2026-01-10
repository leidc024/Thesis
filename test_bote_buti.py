"""
Test disambiguator on bote/buti ambiguous pair
Compares Graph-based Disambiguation vs MaBaybay Default (First Candidate)
Testing with 100 sentences (50 each)
"""

import json
from src.disambiguator import BaybayinDisambiguator

# Read the sentences from gold standard dataset
SENTENCE_FILE = "gold_standard_dataset/sentences/02_bote_buti.txt"

def parse_sentence_file(filepath):
    """Parse sentence file with format: bote/buti sections"""
    bote_sentences = []
    buti_sentences = []
    current_section = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if "BOTE" in line and "bottle" in line:
                    current_section = "bote"
                elif "BUTI" in line and "goodness" in line:
                    current_section = "buti"
                continue
            
            if current_section == "bote":
                bote_sentences.append(line)
            elif current_section == "buti":
                buti_sentences.append(line)
    
    return bote_sentences, buti_sentences

bote_sentences, buti_sentences = parse_sentence_file(SENTENCE_FILE)

print(f"="*70)
print("BOTE/BUTI DISAMBIGUATION TEST")
print("Comparing: Graph-based vs MaBaybay Default (First Candidate)")
print(f"="*70)
print(f"\nLoaded from: {SENTENCE_FILE}")

print(f"\nBote sentences: {len(bote_sentences)}")
print(f"Buti sentences: {len(buti_sentences)}")

# Create test data with OCR candidates
# For bote/buti, both map to Baybayin ᜊᜓᜆᜒ
# MaBaybay default order: ["bote", "buti"] (bote is first candidate)
test_data = []

# Add bote sentences (ground truth = bote)
for sent in bote_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        if word.lower() == "bote":
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
        if word.lower() == "buti":
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
    gt_words = gt.lower().split()
    
    for word in gt_words:
        if word in ['bote', 'buti']:
            # Baseline always picks "bote" (first candidate)
            if word == 'bote':
                baseline_correct_total += 1
                baseline_correct_bote += 1
            # If ground truth is "buti", baseline gets it wrong
            break

baseline_accuracy = baseline_correct_total / len(test_data) * 100

print(f"\nBaseline Strategy: Always select 'bote' (first candidate)")
print(f"Bote accuracy: {baseline_correct_bote}/{len(bote_sentences)} = {baseline_correct_bote/len(bote_sentences):.2%}")
print(f"Buti accuracy: {baseline_correct_buti}/{len(buti_sentences)} = {baseline_correct_buti/len(buti_sentences):.2%}")
print(f"Overall baseline accuracy: {baseline_correct_total}/{len(test_data)} = {baseline_accuracy:.2f}%")

# ============================================================================
# GRAPH-BASED DISAMBIGUATION
# ============================================================================
print("\n" + "="*70)
print("GRAPH-BASED DISAMBIGUATION MODEL")
print("="*70)

all_test_sentences = [item['ground_truth'] for item in test_data]
model = BaybayinDisambiguator(
    corpus_files=[
        "Tagalog_Literary_Text.txt",
        "Tagalog_Religious_Text.txt"
    ],
    exclude_sentences=all_test_sentences  # Clean evaluation - no data leakage
)

# Run evaluation
print("\nRunning disambiguation on test sentences...")

metrics, results = model.evaluate(test_data, show_progress=True)

# Display results
print("\n" + "="*70)
print("GRAPH-BASED DISAMBIGUATION RESULTS")
print("="*70)

print(f"\nAmbiguous words (bote/buti): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Graph-based accuracy: {metrics['ambiguous_accuracy']:.2%} ★")

# Show some examples
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

# Show first 5 correct for each word and first 5 incorrect
correct_bote_examples = []
correct_buti_examples = []
incorrect_examples = []

for i, (test_item, result_item) in enumerate(zip(test_data, results)):
    gt = test_item['ground_truth']
    pred = result_item['predicted']
    
    # Find the ambiguous word
    gt_words = gt.lower().split()
    pred_words = pred.lower().split()
    
    for j, (gt_word, pred_word) in enumerate(zip(gt_words, pred_words)):
        if gt_word in ['bote', 'buti']:
            if gt_word == pred_word:
                if gt_word == 'bote' and len(correct_bote_examples) < 5:
                    correct_bote_examples.append((gt, pred, gt_word, pred_word))
                elif gt_word == 'buti' and len(correct_buti_examples) < 5:
                    correct_buti_examples.append((gt, pred, gt_word, pred_word))
            else:
                if len(incorrect_examples) < 5:
                    incorrect_examples.append((gt, pred, gt_word, pred_word))
            break

print("\n✓ CORRECT BOTE PREDICTIONS (Sample):")
for i, (gt, pred, gt_word, pred_word) in enumerate(correct_bote_examples, 1):
    print(f"\n{i}. Ground Truth: {gt}")
    print(f"   Predicted:    {pred}")
    print(f"   ✓ Correct: {pred_word}")

print("\n✓ CORRECT BUTI PREDICTIONS (Sample):")
for i, (gt, pred, gt_word, pred_word) in enumerate(correct_buti_examples, 1):
    print(f"\n{i}. Ground Truth: {gt}")
    print(f"   Predicted:    {pred}")
    print(f"   ✓ Correct: {pred_word}")

if incorrect_examples:
    print("\n✗ INCORRECT PREDICTIONS (Sample):")
    for i, (gt, pred, gt_word, pred_word) in enumerate(incorrect_examples, 1):
        print(f"\n{i}. Ground Truth: {gt}")
        print(f"   Predicted:    {pred}")
        print(f"   ✗ Expected '{gt_word}' but got '{pred_word}'")

# Breakdown by word type
bote_correct = sum(1 for test_item, result_item in zip(test_data, results)
                   if "bote" in test_item['ground_truth'].lower().split()
                   and "bote" in result_item['predicted'].lower().split())
buti_correct = sum(1 for test_item, result_item in zip(test_data, results)
                   if "buti" in test_item['ground_truth'].lower().split()
                   and "buti" in result_item['predicted'].lower().split())

print("\n" + "="*70)
print("BREAKDOWN BY WORD")
print("="*70)
print(f"\nBote accuracy: {bote_correct}/{len(bote_sentences)} = {bote_correct/len(bote_sentences):.2%}")
print(f"Buti accuracy: {buti_correct}/{len(buti_sentences)} = {buti_correct/len(buti_sentences):.2%}")

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
│ Graph-based             │    {graph_accuracy:6.2f}%       │ Uses context to       │
│ Disambiguation          │                  │ choose correct word   │
├─────────────────────────┼──────────────────┼───────────────────────┤
│ ★ Improvement           │   +{improvement:6.2f}%       │                       │
└─────────────────────────┴──────────────────┴───────────────────────┘

Breakdown by Word:
  • Bote sentences: Graph={bote_correct}/{len(bote_sentences)} vs Baseline={baseline_correct_bote}/{len(bote_sentences)}
  • Buti sentences: Graph={buti_correct}/{len(buti_sentences)} vs Baseline={baseline_correct_buti}/{len(buti_sentences)}

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
            'bote_accuracy': f"{baseline_correct_bote}/{len(bote_sentences)}",
            'buti_accuracy': f"{baseline_correct_buti}/{len(buti_sentences)}"
        },
        'graph_based': {
            'name': 'Graph-based Disambiguation',
            'strategy': 'Context-aware scoring using semantic, frequency, co-occurrence, morphology',
            'accuracy': graph_accuracy,
            'correct': metrics['correct_ambiguous'],
            'bote_accuracy': f"{bote_correct}/{len(bote_sentences)}",
            'buti_accuracy': f"{buti_correct}/{len(buti_sentences)}"
        },
        'improvement': improvement
    },
    'metrics': metrics
}

with open("results_bote_buti.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Detailed results saved to: results_bote_buti.json")
print("="*70)
