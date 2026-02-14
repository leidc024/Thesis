"""
Generate OCR Results JSON for Context-Aware Baybayin Transliteration
This script generates candidate words based on Baybayin ambiguities,
filtered by the Filipino word corpus.
"""

import json
import csv
import os

# Configuration
SENTENCES_FILE = "dataset/processed/Filipino_sentences_500.txt"
OCR_RESULTS_FILE = "dataset/processed/candidates_results.json"
WORD_CORPUS_FILE = "MaBaybay-OCR/Filipino Word Corpus/Tagalog_words_74419+.csv"

# Load Filipino word corpus
def load_word_corpus():
    """Load the Filipino word corpus into a set for fast lookup."""
    words = set()
    try:
        with open(WORD_CORPUS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if present
            for row in reader:
                if row:  # Check if row is not empty
                    word = row[0].strip().lower()
                    words.add(word)
        print(f"✓ Loaded {len(words)} words from corpus")
    except FileNotFoundError:
        print(f"WARNING: Word corpus not found at {WORD_CORPUS_FILE}")
        print("Proceeding without corpus validation...")
        return None
    return words

# Load corpus once at module level
FILIPINO_WORDS = load_word_corpus()

def simulate_ocr_candidates(baybayin_text, latin_text):
    """
    Generates candidate words based on common Baybayin ambiguities:
    - e/i confusion (both map to ᜁ)
    - o/u confusion (both map to ᜂ)
    - d/r confusion (both map to ᜇ)
    
    Only includes candidates that exist in the Filipino word corpus.
    """
    from itertools import product
    
    words = latin_text.split()
    candidates_list = []
    
    for word in words:
        word_lower = word.lower()
        
        # Generate all possible combinations by treating each ambiguous character
        # For each position, create choices: original or alternative
        char_choices = []
        for char in word_lower:
            if char == 'e':
                char_choices.append(['e', 'i'])
            elif char == 'i':
                char_choices.append(['i', 'e'])
            elif char == 'o':
                char_choices.append(['o', 'u'])
            elif char == 'u':
                char_choices.append(['u', 'o'])
            elif char == 'd':
                char_choices.append(['d', 'r'])
            elif char == 'r':
                char_choices.append(['r', 'd'])
            else:
                char_choices.append([char])
        
        # Generate all combinations
        alternatives = set()
        for combo in product(*char_choices):
            alternatives.add(''.join(combo))
        
        # Convert back to list
        alternatives = list(alternatives)
        
        # Filter alternatives to only include words in the corpus
        if FILIPINO_WORDS is not None:
            valid_alternatives = [alt for alt in alternatives if alt.lower() in FILIPINO_WORDS]
            
            # If filtering removes all alternatives, keep at least the original word
            if not valid_alternatives:
                valid_alternatives = [word_lower]
                
            alternatives = valid_alternatives
        
        # Remove duplicates while preserving order
        seen = set()
        alternatives = [x for x in alternatives if not (x.lower() in seen or seen.add(x.lower()))]
        
        # Sort to have consistent ordering (optional, for readability)
        alternatives.sort()
        
        # If only one candidate (unambiguous), just use the word
        # If multiple candidates (ambiguous), use the list
        if len(alternatives) == 1:
            candidates_list.append(alternatives[0])
        else:
            candidates_list.append(alternatives)
    
    return candidates_list

def generate_ocr_results():
    """
    Reads filipino_sentences_v1.txt and generates ocr_results.json with candidate words.
    """
    
    if not os.path.exists(SENTENCES_FILE):
        print(f"ERROR: {SENTENCES_FILE} not found!")
        print("Please run the sentence extraction scripts first.")
        return
    
    ocr_results = []
    
    print("Generating OCR results JSON...")
    print()
    
    # Read sentences from text file
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    for i, sentence in enumerate(sentences, 1):
        # Generate candidate words
        candidates = simulate_ocr_candidates("", sentence)
        
        # Create entry
        entry = {
            "sentence_id": i,
            "ground_truth": sentence,
            "ocr_candidates": candidates
        }
        
        ocr_results.append(entry)
        
        # Print progress every 100 sentences
        if i % 100 == 0:
            print(f"Processed {i}/{len(sentences)} sentences...")
    
    # Save to JSON file
    with open(OCR_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"✓ Results saved to '{OCR_RESULTS_FILE}'")
    print(f"✓ Total sentences processed: {len(ocr_results)}")
    
    # Show a few examples
    print()
    print("=== Sample Results ===")
    for entry in ocr_results[:3]:
        print(f"Sentence {entry['sentence_id']}: {entry['ground_truth']}")
        print(f"  Candidates: {entry['ocr_candidates']}")
        print()

if __name__ == "__main__":
    generate_ocr_results()
