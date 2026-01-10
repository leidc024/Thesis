"""
Script 1: Ambiguous Pair Discovery
Analyzes the Filipino word corpus to discover ALL ambiguous word pairs
that share the same Baybayin representation.

This script is essential for identifying the complete set of E/I, O/U, and D/R
confusions that need to be represented in the evaluation dataset.

UPDATED: Now filters out false positives (same word repeated multiple times)
to focus only on genuinely ambiguous patterns with different words.
UPDATED: Added dictionary filtering to validate Filipino words.
"""

import csv
from collections import defaultdict
import json
from pathlib import Path

# Configuration
WORD_CORPUS_FILE = "MaBaybay-OCR/Filipino Word Corpus/Tagalog_words_74419+.csv"
DICTIONARY_FILE = "tagalog_dict.txt"  # NEW: Dictionary file
OUTPUT_DIR = "dataset/analysis"
OUTPUT_FILE = "ambiguous_pairs_complete.csv"
OUTPUT_JSON = "ambiguous_pairs_complete.json"
STATS_FILE = "ambiguity_statistics.txt"


def latin_to_baybayin(text):
    """
    Converts Latin script Filipino text to Baybayin script.
    Simplified version focusing on ambiguity detection.
    """
    baybayin_chars = {
        # Independent vowels
        'a': 'ᜀ', 'e': 'ᜁ', 'i': 'ᜁ', 'o': 'ᜂ', 'u': 'ᜂ',
        
        # Consonants with 'a' (inherent vowel)
        'ka': 'ᜃ', 'ga': 'ᜄ', 'nga': 'ᜅ',
        'ta': 'ᜆ', 'da': 'ᜇ', 'ra': 'ᜇ', 'na': 'ᜈ',
        'pa': 'ᜉ', 'ba': 'ᜊ', 'ma': 'ᜋ',
        'ya': 'ᜌ', 'la': 'ᜎ', 'wa': 'ᜏ',
        'sa': 'ᜐ', 'ha': 'ᜑ',
        
        # Consonants with 'i' or 'e'
        'ki': 'ᜃᜒ', 'ke': 'ᜃᜒ', 'gi': 'ᜄᜒ', 'ge': 'ᜄᜒ',
        'ngi': 'ᜅᜒ', 'nge': 'ᜅᜒ', 'ti': 'ᜆᜒ', 'te': 'ᜆᜒ',
        'di': 'ᜇᜒ', 'de': 'ᜇᜒ', 'ri': 'ᜇᜒ', 're': 'ᜇᜒ',
        'ni': 'ᜈᜒ', 'ne': 'ᜈᜒ', 'pi': 'ᜉᜒ', 'pe': 'ᜉᜒ',
        'bi': 'ᜊᜒ', 'be': 'ᜊᜒ', 'mi': 'ᜋᜒ', 'me': 'ᜋᜒ',
        'yi': 'ᜌᜒ', 'ye': 'ᜌᜒ', 'li': 'ᜎᜒ', 'le': 'ᜎᜒ',
        'wi': 'ᜏᜒ', 'we': 'ᜏᜒ', 'si': 'ᜐᜒ', 'se': 'ᜐᜒ',
        'hi': 'ᜑᜒ', 'he': 'ᜑᜒ',
        
        # Consonants with 'u' or 'o'
        'ku': 'ᜃᜓ', 'ko': 'ᜃᜓ', 'gu': 'ᜄᜓ', 'go': 'ᜄᜓ',
        'ngu': 'ᜅᜓ', 'ngo': 'ᜅᜓ', 'tu': 'ᜆᜓ', 'to': 'ᜆᜓ',
        'du': 'ᜇᜓ', 'do': 'ᜇᜓ', 'ru': 'ᜇᜓ', 'ro': 'ᜇᜓ',
        'nu': 'ᜈᜓ', 'no': 'ᜈᜓ', 'pu': 'ᜉᜓ', 'po': 'ᜉᜓ',
        'bu': 'ᜊᜓ', 'bo': 'ᜊᜓ', 'mu': 'ᜋᜓ', 'mo': 'ᜋᜓ',
        'yu': 'ᜌᜓ', 'yo': 'ᜌᜓ', 'lu': 'ᜎᜓ', 'lo': 'ᜎᜓ',
        'wu': 'ᜏᜓ', 'wo': 'ᜏᜓ', 'su': 'ᜐᜓ', 'so': 'ᜐᜓ',
        'hu': 'ᜑᜓ', 'ho': 'ᜑᜓ',
        
        # Consonants with virama (cancels inherent vowel)
        'k': 'ᜃ᜔', 'g': 'ᜄ᜔', 'ng': 'ᜅ᜔',
        't': 'ᜆ᜔', 'd': 'ᜇ᜔', 'r': 'ᜇ᜔', 'n': 'ᜈ᜔',
        'p': 'ᜉ᜔', 'b': 'ᜊ᜔', 'm': 'ᜋ᜔',
        'y': 'ᜌ᜔', 'l': 'ᜎ᜔', 'w': 'ᜏ᜔',
        's': 'ᜐ᜔', 'h': 'ᜑ᜔',
    }
    
    text = text.lower()
    result = []
    i = 0
    
    while i < len(text):
        if text[i] == ' ':
            result.append(' ')
            i += 1
            continue
        
        matched = False
        
        # Try 3-character match (for 'nga', 'ngi', etc.)
        if i + 2 < len(text):
            three_char = text[i:i+3]
            if three_char in baybayin_chars:
                result.append(baybayin_chars[three_char])
                i += 3
                matched = True
                continue
        
        # Try 2-character match
        if i + 1 < len(text):
            two_char = text[i:i+2]
            if two_char in baybayin_chars:
                result.append(baybayin_chars[two_char])
                i += 2
                matched = True
                continue
        
        # Try 1-character match
        one_char = text[i]
        if one_char in baybayin_chars:
            result.append(baybayin_chars[one_char])
            i += 1
            matched = True
        else:
            # Keep original if no match (numbers, punctuation)
            result.append(one_char)
            i += 1
    
    return ''.join(result)


def load_dictionary():
    """
    Load Filipino dictionary from text file.
    Returns a set of valid Filipino words (lowercase).
    """
    dictionary = set()
    
    try:
        with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.isdigit():  # Skip empty lines and numbers
                    dictionary.add(word)
        
        print(f"✓ Loaded {len(dictionary)} words from dictionary: {DICTIONARY_FILE}")
        return dictionary
        
    except FileNotFoundError:
        print(f"WARNING: Dictionary file not found at {DICTIONARY_FILE}")
        print("Proceeding without dictionary filtering...")
        return None
    except Exception as e:
        print(f"ERROR loading dictionary: {e}")
        print("Proceeding without dictionary filtering...")
        return None


def classify_ambiguity_type(words):
    """
    Determines what type of ambiguity causes these words to map to same Baybayin.
    Returns: 'E/I', 'O/U', 'D/R', 'COMBINED', or 'UNKNOWN'
    """
    has_e_i = False
    has_o_u = False
    has_d_r = False
    
    # Compare all pairs to find differences
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word1 = words[i].lower()
            word2 = words[j].lower()
            
            if len(word1) != len(word2):
                continue
            
            for c1, c2 in zip(word1, word2):
                if c1 != c2:
                    if (c1 in 'ei' and c2 in 'ei') or (c1 in 'ie' and c2 in 'ie'):
                        has_e_i = True
                    elif (c1 in 'ou' and c2 in 'ou') or (c1 in 'uo' and c2 in 'uo'):
                        has_o_u = True
                    elif (c1 in 'dr' and c2 in 'dr') or (c1 in 'rd' and c2 in 'rd'):
                        has_d_r = True
    
    # Determine classification
    types = []
    if has_e_i:
        types.append('E/I')
    if has_o_u:
        types.append('O/U')
    if has_d_r:
        types.append('D/R')
    
    if len(types) == 0:
        return 'UNKNOWN'
    elif len(types) == 1:
        return types[0]
    else:
        return 'COMBINED'


def load_word_corpus():
    """Load Filipino word corpus and identify proper nouns."""
    words = []
    word_forms = {}  # Track all forms of each word (to identify proper-noun-only words)
    
    try:
        with open(WORD_CORPUS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if present
            for row in reader:
                if row:
                    word = row[0].strip()
                    if word and not word.isdigit():  # Skip empty and pure numbers
                        lower_word = word.lower()
                        if lower_word not in word_forms:
                            word_forms[lower_word] = {'capitalized': 0, 'lowercase': 0}
                        
                        if word[0].isupper():
                            word_forms[lower_word]['capitalized'] += 1
                        else:
                            word_forms[lower_word]['lowercase'] += 1
                        
                        words.append(word)
        
        # Identify proper-noun-only words (only appear capitalized, never lowercase)
        proper_nouns = set()
        for word, counts in word_forms.items():
            if counts['capitalized'] > 0 and counts['lowercase'] == 0:
                proper_nouns.add(word)
        
        print(f"✓ Loaded {len(words)} words from corpus")
        print(f"✓ Identified {len(proper_nouns)} proper-noun-only words to filter")
        
        return words, proper_nouns
        
    except FileNotFoundError:
        print(f"ERROR: Word corpus not found at {WORD_CORPUS_FILE}")
        return [], set()


def find_ambiguous_groups(words, proper_nouns, dictionary=None):
    """
    Groups words by their Baybayin representation.
    Returns dictionary: {baybayin_text: {'unique_words': [...], 'total_occurrences': int, 'valid_words': [...]}}
    Filters out false positives (same word repeated multiple times).
    Filters out proper-noun-only words (like "Norse" that never appear lowercase).
    NEW: Filters using dictionary to keep only valid Filipino words.
    """
    print("\nAnalyzing words for Baybayin ambiguities...")
    baybayin_map = defaultdict(list)
    
    for i, word in enumerate(words):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(words)} words...")
        
        baybayin = latin_to_baybayin(word)
        baybayin_map[baybayin].append(word)
    
    # Filter for ambiguous groups (2+ DIFFERENT words with same Baybayin)
    # Use case-insensitive comparison to avoid false ambiguities like "Kamet" vs "kamit"
    ambiguous_groups = {}
    false_positives = 0
    case_duplicates = 0
    proper_noun_filtered = 0
    no_valid_dict_words = 0
    
    for baybayin, word_list in baybayin_map.items():
        # Count total occurrences in corpus (before filtering)
        total_occurrences = len(word_list)
        
        # Get unique words (case-insensitive) - keep lowercase version
        # BUT filter out proper-noun-only words
        seen_lower = {}
        for word in word_list:
            lower = word.lower()
            # Skip words that only appear as proper nouns (capitalized)
            if lower in proper_nouns:
                continue
            if lower not in seen_lower:
                seen_lower[lower] = word.lower()  # Store lowercase version
        
        unique_words = list(seen_lower.values())
        
        # Count how many were filtered as proper nouns
        all_lower = set(w.lower() for w in word_list)
        filtered_count = len(all_lower) - len(unique_words)
        if filtered_count > 0:
            proper_noun_filtered += filtered_count
        
        # NEW: Dictionary filtering
        # Keep the group if AT LEAST ONE word is valid in dictionary
        valid_words = []
        if dictionary is not None:
            # Check which words are in the dictionary
            for word in unique_words:
                if word.lower() in dictionary:
                    valid_words.append(word)
            
            # If no valid dictionary words found, skip this group entirely
            if len(valid_words) == 0 and len(unique_words) > 0:
                no_valid_dict_words += 1
                continue
            
            # Keep ALL unique words (valid and invalid) if at least one is valid
            # This allows OCR to see invalid outputs like "doktod" alongside valid "doktor"
            # The invalid words are useful as they represent potential OCR errors
        else:
            # If no dictionary, use all unique words
            valid_words = unique_words
        
        if len(valid_words) > 1 or (dictionary is not None and len(valid_words) >= 1 and len(unique_words) > 1):
            # Keep if:
            # 1. Multiple valid words (genuinely ambiguous), OR
            # 2. At least 1 valid word + other candidates (for OCR correction)
            # Store ALL unique words (including invalid ones for OCR training)
            ambiguous_groups[baybayin] = {
                'unique_words': unique_words,  # All words (valid + invalid)
                'total_occurrences': total_occurrences,
                'valid_words': valid_words  # Only dictionary-validated words
            }
        elif len(set(word.lower() for word in word_list)) == 1 and len(set(word_list)) > 1:
            # Same word with different capitalization (e.g., "Kamet" vs "kamit")
            case_duplicates += 1
        elif len(word_list) > 1:
            # False positive - same word repeated multiple times
            false_positives += 1
    
    print(f"✓ Found {len(ambiguous_groups)} genuinely ambiguous Baybayin patterns")
    print(f"✓ Filtered out {false_positives} false positives (duplicate words)")
    print(f"✓ Filtered out {case_duplicates} case-only differences (e.g., 'Kamet' vs 'kamit')")
    print(f"✓ Filtered out {proper_noun_filtered} proper-noun-only words (e.g., 'Norse')")
    if dictionary is not None:
        print(f"✓ Filtered out {no_valid_dict_words} groups with no valid dictionary words")
    print(f"✓ Total unique ambiguous words: {sum(len(g['unique_words']) for g in ambiguous_groups.values())}")
    
    return ambiguous_groups

def generate_statistics(ambiguous_groups):
    """Generate detailed statistics about ambiguities."""
    stats = {
        'total_ambiguous_patterns': len(ambiguous_groups),
        'total_ambiguous_words': sum(len(g['unique_words']) for g in ambiguous_groups.values()),
        'total_occurrences': sum(g['total_occurrences'] for g in ambiguous_groups.values()),
        'by_type': defaultdict(int),
        'by_group_size': defaultdict(int),
        'largest_groups': [],
    }
    
    # Classify by type and count
    for baybayin, group in ambiguous_groups.items():
        words = group['unique_words']
        ambiguity_type = classify_ambiguity_type(words)
        stats['by_type'][ambiguity_type] += 1
        unique_word_count = len(words)
        stats['by_group_size'][unique_word_count] += 1
    
    # Find largest groups (by total occurrences)
    sorted_groups = sorted(ambiguous_groups.items(), key=lambda x: x[1]['total_occurrences'], reverse=True)
    stats['largest_groups'] = [
        {
            'baybayin': baybayin, 
            'unique_words': sorted(group['unique_words']),
            'total_occurrences': group['total_occurrences'],
            'unique_count': len(group['unique_words'])
        }
        for baybayin, group in sorted_groups[:10]
    ]
    
    return stats


def save_results(ambiguous_groups, stats):
    """Save results to CSV and JSON files."""
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    for baybayin, group in ambiguous_groups.items():
        unique_words = group['unique_words']
        total_occurrences = group['total_occurrences']
        ambiguity_type = classify_ambiguity_type(unique_words)
        csv_data.append({
            'baybayin': baybayin,
            'ambiguity_type': ambiguity_type,
            'word_count': total_occurrences,  # Total occurrences in corpus
            'unique_word_count': len(unique_words),  # Unique words
            'words': ', '.join(sorted(unique_words)),
        })
    
    # Sort by word count (descending) then alphabetically
    csv_data.sort(key=lambda x: (-x['word_count'], x['words']))
    
    # Save CSV
    csv_path = output_path / OUTPUT_FILE
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['baybayin', 'ambiguity_type', 'word_count', 'unique_word_count', 'words'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n✓ Saved CSV to: {csv_path}")
    
    # Save JSON (more detailed)
    json_data = {
        'metadata': {
            'total_patterns': len(ambiguous_groups),
            'total_unique_words': sum(len(g['unique_words']) for g in ambiguous_groups.values()),
            'total_occurrences': sum(g['total_occurrences'] for g in ambiguous_groups.values()),
            'source_corpus': WORD_CORPUS_FILE,
            'dictionary_file': DICTIONARY_FILE,
        },
        'statistics': dict(stats),
        'ambiguous_groups': [
            {
                'baybayin': baybayin,
                'ambiguity_type': classify_ambiguity_type(group['unique_words']),
                'words': sorted(group['unique_words']),
                'unique_count': len(group['unique_words']),
                'total_occurrences': group['total_occurrences'],
            }
            for baybayin, group in sorted(ambiguous_groups.items(), 
                                         key=lambda x: x[1]['total_occurrences'], 
                                         reverse=True)
        ]
    }
    
    json_path = output_path / OUTPUT_JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved JSON to: {json_path}")
    
    # Save statistics text file
    stats_path = output_path / STATS_FILE
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BAYBAYIN AMBIGUITY ANALYSIS STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total ambiguous Baybayin patterns: {stats['total_ambiguous_patterns']}\n")
        f.write(f"Total unique ambiguous words: {stats['total_ambiguous_words']}\n")
        f.write(f"Total occurrences in corpus: {stats['total_occurrences']}\n\n")
        
        f.write("BREAKDOWN BY AMBIGUITY TYPE:\n")
        f.write("-" * 40 + "\n")
        for amb_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
            f.write(f"  {amb_type:12s}: {count:5d} patterns\n")
        
        f.write("\nBREAKDOWN BY UNIQUE WORD COUNT:\n")
        f.write("-" * 40 + "\n")
        for size, count in sorted(stats['by_group_size'].items()):
            f.write(f"  {size} unique words: {count:5d} patterns\n")
        
        f.write("\nTOP 10 LARGEST AMBIGUOUS GROUPS (by corpus occurrences):\n")
        f.write("-" * 40 + "\n")
        for i, group in enumerate(stats['largest_groups'][:10], 1):
            unique_words = group['unique_words']
            f.write(f"\n{i}. {group['total_occurrences']} occurrences, {group['unique_count']} unique words sharing Baybayin: {group['baybayin']}\n")
            f.write(f"   Unique words: {', '.join(unique_words)}\n")
    
    print(f"✓ Saved statistics to: {stats_path}")


def print_summary(stats):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("AMBIGUITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal ambiguous patterns: {stats['total_ambiguous_patterns']}")
    print(f"Total unique ambiguous words: {stats['total_ambiguous_words']}")
    print(f"Total occurrences in corpus: {stats['total_occurrences']}")
    
    print("\nBy Ambiguity Type:")
    for amb_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        percentage = (count / stats['total_ambiguous_patterns']) * 100
        print(f"  {amb_type:12s}: {count:5d} patterns ({percentage:.1f}%)")
    
    print("\nTop 5 Largest Ambiguous Groups (by corpus occurrences):")
    for i, group in enumerate(stats['largest_groups'][:5], 1):
        unique_words = group['unique_words']
        print(f"  {i}. {group['total_occurrences']} occurrences, {group['unique_count']} unique words sharing Baybayin: {group['baybayin']}")
        print(f"     Unique words: {', '.join(unique_words)}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("BAYBAYIN AMBIGUOUS PAIR DISCOVERY")
    print("=" * 70)
    
    # Load dictionary
    dictionary = load_dictionary()
    
    # Load corpus
    result = load_word_corpus()
    if not result or not result[0]:
        return
    words, proper_nouns = result
    
    # Find ambiguous groups (with dictionary filtering)
    ambiguous_groups = find_ambiguous_groups(words, proper_nouns, dictionary)
    
    # Generate statistics
    stats = generate_statistics(ambiguous_groups)
    
    # Save results
    save_results(ambiguous_groups, stats)
    
    # Print summary
    print_summary(stats)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review ambiguous_pairs_complete.csv for all discovered pairs")
    print("2. Identify high-frequency pairs for priority inclusion in dataset")
    print("3. Run sentence extraction script (02_extract_sentences.py)")


if __name__ == "__main__":
    main()