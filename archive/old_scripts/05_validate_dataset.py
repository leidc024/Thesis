"""
Script 5: Dataset Validator
Performs comprehensive quality assurance checks on the final dataset.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

# Configuration
SENTENCES_FILE = "dataset/processed/filipino_sentences_v1.txt"
ANNOTATIONS_FILE = "dataset/processed/annotations_v1.csv"
CANDIDATES_FILE = "dataset/processed/candidates_results_v1.json"
WORD_CORPUS_FILE = "MaBaybay-OCR/Filipino Word Corpus/Tagalog_words_74419+.csv"
AMBIGUOUS_PAIRS_FILE = "dataset/analysis/ambiguous_pairs_complete.json"
OUTPUT_FILE = "dataset/analysis/validation_report.json"

# Targets for validation
TARGETS = {
    'total_min': 500,
    'total_max': 1000,
    'ambiguity_types': {
        'E/I': (0.30, 0.40),  # 30-40%
        'O/U': (0.30, 0.40),
        'D/R': (0.10, 0.20),
        'COMBINED': (0.05, 0.15),
    },
    'density_levels': {
        'low': (0.35, 0.45),
        'medium': (0.35, 0.45),
        'high': (0.15, 0.25),
    },
    'difficulty_levels': {
        'easy': (0.25, 0.35),
        'medium': (0.45, 0.55),
        'hard': (0.15, 0.25),
    },
    'min_examples_per_pair': 3,
}


def load_word_corpus():
    """Load the Filipino word corpus for validation."""
    words = set()
    try:
        with open(WORD_CORPUS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    words.add(row[0].strip().lower())
        print(f"✓ Loaded {len(words)} words from corpus")
        return words
    except FileNotFoundError:
        print(f"WARNING: Could not load word corpus")
        return None


def load_sentences():
    """Load the final sentence dataset."""
    try:
        with open(SENTENCES_FILE, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(sentences)} sentences")
        return sentences
    except FileNotFoundError:
        print(f"ERROR: Sentences file not found: {SENTENCES_FILE}")
        return []


def load_annotations():
    """Load annotations with ground truth."""
    try:
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            annotations = list(reader)
        print(f"✓ Loaded {len(annotations)} annotations")
        return annotations
    except FileNotFoundError:
        print(f"ERROR: Annotations file not found: {ANNOTATIONS_FILE}")
        return []


def load_candidates():
    """Load OCR candidates."""
    try:
        with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        print(f"✓ Loaded {len(candidates)} candidate results")
        return candidates
    except FileNotFoundError:
        print(f"ERROR: Candidates file not found: {CANDIDATES_FILE}")
        return []


def validate_size(sentences):
    """Check if dataset size meets targets."""
    count = len(sentences)
    passed = TARGETS['total_min'] <= count <= TARGETS['total_max']
    
    return {
        'test': 'Dataset Size',
        'passed': passed,
        'value': count,
        'target': f"{TARGETS['total_min']}-{TARGETS['total_max']}",
        'message': f"Dataset has {count} sentences" + 
                  (" ✓" if passed else f" (target: {TARGETS['total_min']}-{TARGETS['total_max']})"),
    }


def validate_duplicates(sentences):
    """Check for duplicate sentences."""
    seen = set()
    duplicates = []
    
    for i, sentence in enumerate(sentences):
        normalized = sentence.lower().strip()
        if normalized in seen:
            duplicates.append((i, sentence))
        seen.add(normalized)
    
    passed = len(duplicates) == 0
    
    return {
        'test': 'No Duplicates',
        'passed': passed,
        'value': len(duplicates),
        'target': '0',
        'message': f"Found {len(duplicates)} duplicate sentences" + (" ✓" if passed else " ✗"),
        'details': duplicates[:5] if duplicates else None,
    }


def validate_corpus_words(candidates, word_corpus):
    """Validate that all words exist in the Filipino corpus."""
    if word_corpus is None:
        return {
            'test': 'Corpus Word Validation',
            'passed': None,
            'message': 'Skipped (corpus not available)',
        }
    
    invalid_words = []
    total_words = 0
    
    for entry in candidates:
        for candidate in entry['ocr_candidates']:
            if isinstance(candidate, list):
                words_to_check = candidate
            else:
                words_to_check = [candidate]
            
            for word in words_to_check:
                total_words += 1
                if word.lower() not in word_corpus:
                    invalid_words.append(word)
    
    passed = len(invalid_words) == 0
    invalid_rate = len(invalid_words) / total_words if total_words > 0 else 0
    
    return {
        'test': 'Corpus Word Validation',
        'passed': passed,
        'value': len(invalid_words),
        'total_words': total_words,
        'invalid_rate': f"{invalid_rate*100:.2f}%",
        'target': '100% valid',
        'message': f"{len(invalid_words)}/{total_words} words not in corpus ({invalid_rate*100:.2f}%)" +
                  (" ✓" if passed else " ✗"),
        'details': list(set(invalid_words))[:10] if invalid_words else None,
    }


def validate_distribution(candidates, target_dict, key_name, extract_func):
    """Generic distribution validation."""
    if not candidates:
        return {
            'test': f'{key_name} Distribution',
            'passed': None,
            'message': 'No candidate data available',
        }
    
    # Count occurrences
    counts = Counter()
    for entry in candidates:
        value = extract_func(entry)
        if value:
            counts[value] += 1
    
    total = sum(counts.values())
    if total == 0:
        return {
            'test': f'{key_name} Distribution',
            'passed': False,
            'message': 'No data to validate',
        }
    
    # Check each target
    results = {}
    all_passed = True
    
    for category, (min_pct, max_pct) in target_dict.items():
        count = counts.get(category, 0)
        actual_pct = count / total
        passed = min_pct <= actual_pct <= max_pct
        all_passed = all_passed and passed
        
        results[category] = {
            'count': count,
            'percentage': f"{actual_pct*100:.1f}%",
            'target': f"{min_pct*100:.0f}-{max_pct*100:.0f}%",
            'passed': passed,
        }
    
    return {
        'test': f'{key_name} Distribution',
        'passed': all_passed,
        'total': total,
        'breakdown': results,
        'message': f"Distribution check {'passed ✓' if all_passed else 'failed ✗'}",
    }


def extract_ambiguity_types(entry):
    """Extract ambiguity types from candidate entry."""
    # This is a simplified extraction - you may need to enhance
    # based on how you structure your candidates file
    types = set()
    for candidate in entry.get('ocr_candidates', []):
        if isinstance(candidate, list) and len(candidate) > 1:
            # Try to infer type from differences
            word1 = candidate[0].lower()
            word2 = candidate[1].lower()
            if any(c1 in 'ei' and c2 in 'ei' and c1 != c2 
                  for c1, c2 in zip(word1, word2)):
                types.add('E/I')
            if any(c1 in 'ou' and c2 in 'ou' and c1 != c2 
                  for c1, c2 in zip(word1, word2)):
                types.add('O/U')
            if any(c1 in 'dr' and c2 in 'dr' and c1 != c2 
                  for c1, c2 in zip(word1, word2)):
                types.add('D/R')
    
    if len(types) > 1:
        return 'COMBINED'
    elif len(types) == 1:
        return list(types)[0]
    return None


def validate_coverage(candidates):
    """Validate coverage of ambiguous pairs."""
    pair_counts = defaultdict(int)
    
    for entry in candidates:
        for candidate in entry.get('ocr_candidates', []):
            if isinstance(candidate, list):
                # This is an ambiguous word
                pair_key = tuple(sorted(candidate))
                pair_counts[pair_key] += 1
    
    # Check minimum coverage
    under_represented = {
        pair: count 
        for pair, count in pair_counts.items() 
        if count < TARGETS['min_examples_per_pair']
    }
    
    passed = len(under_represented) == 0
    
    return {
        'test': 'Ambiguous Pair Coverage',
        'passed': passed,
        'total_unique_pairs': len(pair_counts),
        'under_represented': len(under_represented),
        'min_examples_required': TARGETS['min_examples_per_pair'],
        'message': f"{len(under_represented)} pairs with <{TARGETS['min_examples_per_pair']} examples" +
                  (" ✓" if passed else " ✗"),
        'details': dict(list(under_represented.items())[:10]) if under_represented else None,
    }


def validate_length(sentences):
    """Validate sentence lengths."""
    lengths = [len(s.split()) for s in sentences]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    
    passed = min_len >= 5 and max_len <= 20
    
    return {
        'test': 'Sentence Length',
        'passed': passed,
        'min': min_len,
        'max': max_len,
        'average': f"{avg_len:.1f}",
        'target': '5-20 words',
        'message': f"Length range: {min_len}-{max_len} words (avg: {avg_len:.1f})" +
                  (" ✓" if passed else " ✗"),
    }


def generate_statistics(sentences, candidates):
    """Generate comprehensive dataset statistics."""
    stats = {
        'total_sentences': len(sentences),
        'total_words': sum(len(s.split()) for s in sentences),
        'unique_words': len(set(word.lower() for s in sentences for word in s.split())),
        'total_ambiguous_positions': 0,
    }
    
    # Count ambiguous positions
    for entry in candidates:
        for candidate in entry.get('ocr_candidates', []):
            if isinstance(candidate, list):
                stats['total_ambiguous_positions'] += 1
    
    # Calculate ratios
    stats['type_token_ratio'] = stats['unique_words'] / stats['total_words'] if stats['total_words'] > 0 else 0
    stats['ambiguity_density'] = stats['total_ambiguous_positions'] / stats['total_sentences'] if stats['total_sentences'] > 0 else 0
    
    return stats


def run_validation():
    """Run all validation tests."""
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)
    print()
    
    # Load data
    word_corpus = load_word_corpus()
    sentences = load_sentences()
    annotations = load_annotations()
    candidates = load_candidates()
    
    if not sentences:
        print("ERROR: Cannot validate without sentences")
        return
    
    print()
    print("Running validation tests...")
    print("-" * 70)
    
    # Run tests
    tests = []
    
    tests.append(validate_size(sentences))
    tests.append(validate_duplicates(sentences))
    tests.append(validate_length(sentences))
    tests.append(validate_corpus_words(candidates, word_corpus))
    tests.append(validate_coverage(candidates))
    
    # Print results
    print()
    for test in tests:
        status = "✓ PASS" if test['passed'] else ("✗ FAIL" if test['passed'] is False else "⊘ SKIP")
        print(f"{status:8s} | {test['test']}")
        print(f"         | {test['message']}")
        if test.get('details'):
            print(f"         | Details: {test['details']}")
        print()
    
    # Generate statistics
    stats = generate_statistics(sentences, candidates)
    
    # Save report
    report = {
        'validation_date': '2025-12-02',
        'tests': tests,
        'statistics': stats,
        'summary': {
            'total_tests': len([t for t in tests if t['passed'] is not None]),
            'passed': len([t for t in tests if t['passed'] is True]),
            'failed': len([t for t in tests if t['passed'] is False]),
            'skipped': len([t for t in tests if t['passed'] is None]),
        }
    }
    
    # Save report
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Skipped: {report['summary']['skipped']}")
    print()
    print(f"Report saved to: {output_path}")
    
    # Overall status
    if report['summary']['failed'] == 0:
        print("\n✓ DATASET VALIDATION PASSED - Ready for use!")
    else:
        print(f"\n✗ DATASET VALIDATION FAILED - {report['summary']['failed']} test(s) failed")
        print("Review the report and address issues before proceeding.")


if __name__ == "__main__":
    run_validation()
