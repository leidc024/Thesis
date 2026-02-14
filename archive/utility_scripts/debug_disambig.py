"""Debug disambiguation scoring"""
import sys
import io

# Suppress init output
old_stdout = sys.stdout
sys.stdout = io.StringIO()

from src.disambiguator import BaybayinDisambiguator

model = BaybayinDisambiguator(
    corpus_files=['Tagalog_Literary_Text.txt', 'Tagalog_Religious_Text.txt']
)

sys.stdout = old_stdout

# Test with 4 candidates
candidates = [['doktod', 'doktor', 'duktod', 'duktor']]

print("=" * 50)
print("Testing disambiguation for:", candidates)
print("=" * 50)

# Check corpus frequency for each
print("\nCorpus frequencies:")
for word in candidates[0]:
    score = model.corpus.get_frequency_score(word)
    count = model.corpus.word_freq.get(word, 0)
    print(f"  {word}: score={score:.4f}, count={count}")

# Run disambiguation with detailed output
result, debug = model.disambiguate(candidates)

print(f"\nResult: {result}")
print(f"\nDebug info:")
for key, value in debug.items():
    print(f"  {key}: {value}")
