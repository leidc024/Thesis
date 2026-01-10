"""
Script 3: Dataset Balancer
Creates a balanced final dataset from candidate sentences through stratified sampling.
Aims to meet distribution targets for ambiguity types and difficulty levels.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Configuration
CANDIDATES_FILE = "dataset/raw/candidate_sentences.json"
OUTPUT_FILE = "dataset/processed/filipino_sentences_v1.txt"
STATS_FILE = "dataset/analysis/dataset_statistics.json"

# Target dataset size
TARGET_SIZE = 1000  # 1000 for publication quality

# Distribution targets (percentages)
# Note: UNKNOWN removed since those were false ambiguities (capitalization only)
# The 5% is redistributed to E/I and O/U
TARGETS = {
    'E/I': 0.375,     # 37.5% = 188 sentences
    'O/U': 0.375,     # 37.5% = 187 sentences  
    'D/R': 0.15,      # 15% = 75 sentences
    'COMBINED': 0.10, # 10% = 50 sentences
}
# Total: 188 + 187 + 75 + 50 = 500

# Seed for reproducibility
RANDOM_SEED = 42


def load_candidates():
    """Load candidate sentences from extraction phase."""
    try:
        with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data['candidates'])} candidate sentences")
        return data['candidates']
    except FileNotFoundError:
        print(f"ERROR: Candidates file not found at {CANDIDATES_FILE}")
        print("Please run 02_extract_sentences.py first!")
        return []


def classify_sentence(candidate):
    """
    Classify sentence by primary ambiguity type.
    Returns the most significant ambiguity type found.
    """
    types = candidate.get('ambiguity_types', [])
    
    # Priority order: COMBINED > specific types > UNKNOWN
    if 'COMBINED' in types:
        return 'COMBINED'
    elif 'E/I' in types:
        return 'E/I'
    elif 'O/U' in types:
        return 'O/U'
    elif 'D/R' in types:
        return 'D/R'
    else:
        return 'UNKNOWN'


def clean_sentence(sentence):
    """Clean up sentence text (remove titles, extra spaces, etc.)."""
    # Remove common prefixes
    sentence = sentence.replace('Title:', '')
    sentence = sentence.replace('Text', '')
    
    # Remove multiple spaces
    import re
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # Remove sentences that start with numbers or are too short
    if sentence and not sentence[0].isdigit() and len(sentence) > 20:
        return sentence
    return None


def filter_quality(candidates):
    """Filter candidates for quality (proper sentences only)."""
    quality = []
    
    for candidate in candidates:
        sentence = candidate['sentence']
        
        # Skip if it looks like metadata or titles
        if any(x in sentence.lower() for x in ['title:', 'word count:', 'text ', '<b>', 'http']):
            continue
        
        # Skip very short or very long sentences
        word_count = candidate['word_count']
        if word_count < 5 or word_count > 10:
            continue
        
        # Clean the sentence
        cleaned = clean_sentence(sentence)
        if cleaned:
            candidate['sentence'] = cleaned
            quality.append(candidate)
    
    print(f"✓ Filtered to {len(quality)} quality sentences")
    return quality


def stratified_sample(candidates, target_size):
    """
    Perform stratified random sampling to meet distribution targets.
    """
    random.seed(RANDOM_SEED)
    
    # Group by ambiguity type
    by_type = defaultdict(list)
    for candidate in candidates:
        ambiguity_type = classify_sentence(candidate)
        by_type[ambiguity_type].append(candidate)
    
    print("\nAvailable by type:")
    for amb_type, items in sorted(by_type.items()):
        print(f"  {amb_type:12s}: {len(items):5d} candidates")
    
    # Calculate target counts - use exact numbers to reach 1000
    target_counts = {
        'E/I': 375,      # 37.5%
        'O/U': 375,      # 37.5%
        'D/R': 150,      # 15%
        'COMBINED': 100, # 10%
    }
    # Total: 375 + 375 + 150 + 100 = 1000
    
    print("\nTarget counts:")
    for amb_type, count in sorted(target_counts.items()):
        print(f"  {amb_type:12s}: {count:5d} sentences ({count/target_size*100:.0f}%)")
    
    # Sample from each type
    selected = []
    seen_sentences = set()  # Track unique sentences to avoid duplicates
    
    for amb_type, target_count in target_counts.items():
        available = by_type.get(amb_type, [])
        
        # Filter out sentences we've already selected
        available = [c for c in available if c['sentence'] not in seen_sentences]
        
        if len(available) >= target_count:
            # Enough candidates, sample randomly
            sampled = random.sample(available, target_count)
        else:
            # Not enough candidates, take all
            sampled = available
            shortage = target_count - len(available)
            print(f"\n⚠ Only {len(available)} {amb_type} sentences available (need {target_count})")
            print(f"  Shortage: {shortage} sentences")
        
        # Add to selected and mark as seen
        for s in sampled:
            selected.append(s)
            seen_sentences.add(s['sentence'])
        
        print(f"\n✓ Sampled {len(sampled)} {amb_type} sentences")
    
    print(f"\n✓ Total selected: {len(selected)} sentences (all unique)")
    return selected


def generate_statistics(selected):
    """Generate comprehensive statistics about the final dataset."""
    stats = {
        'total_sentences': len(selected),
        'by_type': defaultdict(int),
        'by_density': defaultdict(int),
        'by_difficulty': defaultdict(int),
        'by_source': defaultdict(int),
        'word_count_stats': {
            'min': min(c['word_count'] for c in selected),
            'max': max(c['word_count'] for c in selected),
            'avg': sum(c['word_count'] for c in selected) / len(selected),
        },
        'total_ambiguous_positions': sum(c['ambiguity_count'] for c in selected),
    }
    
    for candidate in selected:
        amb_type = classify_sentence(candidate)
        stats['by_type'][amb_type] += 1
        stats['by_density'][candidate['density_level']] += 1
        stats['by_difficulty'][candidate['difficulty_estimate']] += 1
        stats['by_source'][candidate['source']] += 1
    
    # Convert to percentages
    total = len(selected)
    stats['by_type_pct'] = {k: (v/total)*100 for k, v in stats['by_type'].items()}
    stats['by_density_pct'] = {k: (v/total)*100 for k, v in stats['by_density'].items()}
    stats['by_difficulty_pct'] = {k: (v/total)*100 for k, v in stats['by_difficulty'].items()}
    
    return stats


def save_results(selected, stats):
    """Save the final dataset and statistics."""
    # Create output directory
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save sentences (one per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for candidate in selected:
            f.write(candidate['sentence'] + '\n')
    
    print(f"\n✓ Saved {len(selected)} sentences to: {output_path}")
    
    # Save statistics
    stats_path = Path(STATS_FILE)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✓ Saved statistics to: {stats_path}")


def print_summary(stats):
    """Print dataset summary."""
    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal sentences: {stats['total_sentences']}")
    print(f"Total ambiguous positions: {stats['total_ambiguous_positions']}")
    print(f"Avg ambiguous words per sentence: {stats['total_ambiguous_positions']/stats['total_sentences']:.1f}")
    
    print("\nBy Ambiguity Type:")
    for amb_type in ['E/I', 'O/U', 'D/R', 'COMBINED', 'UNKNOWN']:
        count = stats['by_type'].get(amb_type, 0)
        pct = stats['by_type_pct'].get(amb_type, 0)
        target_pct = TARGETS.get(amb_type, 0) * 100
        status = "✓" if abs(pct - target_pct) <= 5 else "⚠"
        print(f"  {status} {amb_type:12s}: {count:3d} ({pct:5.1f}% | target: {target_pct:.0f}%)")
    
    print("\nBy Density Level:")
    for level in ['low', 'medium', 'high']:
        count = stats['by_density'].get(level, 0)
        pct = stats['by_density_pct'].get(level, 0)
        print(f"  {level:10s}: {count:3d} ({pct:5.1f}%)")
    
    print("\nBy Difficulty:")
    for difficulty in ['easy', 'medium', 'hard']:
        count = stats['by_difficulty'].get(difficulty, 0)
        pct = stats['by_difficulty_pct'].get(difficulty, 0)
        print(f"  {difficulty:10s}: {count:3d} ({pct:5.1f}%)")
    
    print("\nBy Source:")
    for source, count in sorted(stats['by_source'].items()):
        pct = (count / stats['total_sentences']) * 100
        print(f"  {source:15s}: {count:3d} ({pct:5.1f}%)")
    
    print("\nWord Count Statistics:")
    print(f"  Min: {stats['word_count_stats']['min']} words")
    print(f"  Max: {stats['word_count_stats']['max']} words")
    print(f"  Avg: {stats['word_count_stats']['avg']:.1f} words")


def show_samples(selected):
    """Show sample sentences from each category."""
    print("\n" + "=" * 70)
    print("SAMPLE SENTENCES")
    print("=" * 70)
    
    by_type = defaultdict(list)
    for candidate in selected:
        amb_type = classify_sentence(candidate)
        by_type[amb_type].append(candidate)
    
    for amb_type in ['E/I', 'O/U', 'D/R', 'COMBINED']:
        samples = by_type.get(amb_type, [])[:3]
        if samples:
            print(f"\n{amb_type} Examples:")
            print("-" * 70)
            for i, sample in enumerate(samples, 1):
                print(f"{i}. {sample['sentence']}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("DATASET BALANCING - STRATIFIED SAMPLING")
    print("=" * 70)
    print(f"\nTarget size: {TARGET_SIZE} sentences")
    
    # Load candidates
    candidates = load_candidates()
    if not candidates:
        return
    
    # Filter for quality
    quality_candidates = filter_quality(candidates)
    if not quality_candidates:
        print("ERROR: No quality candidates remaining!")
        return
    
    # Stratified sampling
    selected = stratified_sample(quality_candidates, TARGET_SIZE)
    
    if len(selected) < TARGET_SIZE * 0.8:  # Less than 80% of target
        print(f"\n⚠ WARNING: Only got {len(selected)} sentences (target: {TARGET_SIZE})")
        print("Consider:")
        print("  1. Lowering target size")
        print("  2. Adding more source corpora")
        print("  3. Manually constructing sentences for gaps")
    
    # Generate statistics
    stats = generate_statistics(selected)
    
    # Save results
    save_results(selected, stats)
    
    # Print summary
    print_summary(stats)
    
    # Show samples
    show_samples(selected)
    
    print("\n" + "=" * 70)
    print("BALANCING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review filipino_sentences_v1.txt")
    print("2. Generate Baybayin images (adapt create_dataset.py)")
    print("3. Generate OCR candidates (adapt generate_candidate_results.py)")
    print("4. Run validation (05_validate_dataset.py)")


if __name__ == "__main__":
    main()
