"""
Create a balanced sample of Tagalog_Balita_Texts.json
Target: ~300,000 words (similar to Literary + Religious combined)
Strategy: Prioritize sentences containing the 15 ambiguous word pairs
"""

import json
import re

# Define the 15 ambiguous word pairs
AMBIGUOUS_PAIRS = [
    ['asero', 'asido'],
    ['bote', 'buti'],
    ['boto', 'buto'],
    ['higante', 'higanti'],
    ['hito', 'heto'],
    ['itodo', 'ituro'],
    ['kamada', 'kamara'],
    ['kompas', 'kumpas'],
    ['kumita', 'kometa'],
    ['mesa', 'misa'],
    ['polo', 'pulo'],
    ['poso', 'puso'],
    ['tela', 'tila'],
    ['todo', 'toro', 'turo'],
    ['toyo', 'tuyo']
]

# Flatten to single list for searching
all_ambiguous_words = set()
for pair in AMBIGUOUS_PAIRS:
    all_ambiguous_words.update(pair)

print(f"Targeting {len(all_ambiguous_words)} ambiguous words from 15 pairs")
print(f"Words: {sorted(all_ambiguous_words)}\n")

# Read JSON file
print("Reading Tagalog_Balita_Text.json...")
with open("Tagalog_Balita_Text.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total articles: {len(data)}\n")

# Extract sentences and categorize
priority_sentences = []  # Contains ambiguous words
regular_sentences = []   # Does not contain ambiguous words

for article in data:
    if 'body' in article and article['body']:
        for paragraph in article['body']:
            # Split into sentences
            sentences = re.split(r'[.!?]+', paragraph)
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) < 4:  # Skip short sentences
                    continue
                
                # Check if sentence contains any ambiguous word
                words_lower = set(re.findall(r'\b\w+\b', sent.lower()))
                if words_lower & all_ambiguous_words:
                    priority_sentences.append(sent)
                else:
                    regular_sentences.append(sent)

print(f"Priority sentences (with ambiguous words): {len(priority_sentences):,}")
print(f"Regular sentences: {len(regular_sentences):,}")

# Calculate words
priority_words = sum(len(s.split()) for s in priority_sentences)
print(f"Priority words: {priority_words:,}\n")

# Build balanced corpus
target_words = 600000  # Increased from 300k to get better coverage
selected_sentences = []
current_words = 0

# First, add ALL priority sentences (they're valuable for disambiguation!)
print("Adding priority sentences...")
for sent in priority_sentences:
    word_count = len(sent.split())
    if current_words + word_count <= target_words:
        selected_sentences.append(sent)
        current_words += word_count
    else:
        break

print(f"Added {len(selected_sentences):,} priority sentences ({current_words:,} words)")

# Then fill remaining with regular sentences
print("Filling with regular sentences...")
remaining_target = target_words - current_words
added_regular = 0

for sent in regular_sentences:
    word_count = len(sent.split())
    if current_words + word_count <= target_words:
        selected_sentences.append(sent)
        current_words += word_count
        added_regular += 1
    else:
        break
    
    if current_words >= target_words:
        break

print(f"Added {added_regular:,} regular sentences")

# Write balanced corpus
with open("Tagalog_Balita_Texts_Balanced.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(selected_sentences))

print(f"\n✓ Created Tagalog_Balita_Texts_Balanced.txt")
print(f"  Total sentences: {len(selected_sentences):,}")
print(f"  Total words: {current_words:,}")
print(f"\nComparison:")
print(f"  Literary:  203,499 words")
print(f"  Religious:  86,908 words")
print(f"  Balanced:  {current_words:,} words")
print(f"  ─────────────────────────")
print(f"  Total:     ~{203499 + 86908 + current_words:,} words")
print(f"\n✓ Now update disambiguator.py to use 'Tagalog_Balita_Texts_Balanced.txt'")
