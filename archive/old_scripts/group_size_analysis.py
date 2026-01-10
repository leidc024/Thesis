"""
Demonstrate what the group size statistics SHOULD look like
when counting unique words instead of occurrences
"""
import csv

def analyze_group_sizes():
    print("=" * 70)
    print("GROUP SIZE ANALYSIS: Occurrences vs Unique Words")
    print("=" * 70)
    print()
    
    # Read the CSV to get actual unique word counts
    unique_word_counts = {}
    
    with open('dataset/analysis/ambiguous_pairs_complete.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_count = int(row['unique_word_count'])
            if unique_count not in unique_word_counts:
                unique_word_counts[unique_count] = 0
            unique_word_counts[unique_count] += 1
    
    print("📊 BREAKDOWN BY UNIQUE WORD COUNT (What it SHOULD show):")
    print("-" * 50)
    
    total_patterns = 0
    for size in sorted(unique_word_counts.keys()):
        count = unique_word_counts[size]
        total_patterns += count
        print(f"  {size} unique words: {count:5d} patterns")
    
    print(f"\n  Total: {total_patterns} patterns")
    
    print()
    print("=" * 70)
    print("WHAT THIS MEANS:")
    print("=" * 70)
    print()
    
    print("📋 EXAMPLES:")
    print()
    print("  '2 unique words' patterns include:")
    print("     • ᜎᜃᜒ → lake, laki (E/I ambiguity)")
    print("     • ᜊᜓᜆᜒ → bote, buti (Combined ambiguity)")
    print("     • ᜇᜀᜈ᜔ → daan, raan (D/R ambiguity)")
    print()
    print("  '3 unique words' patterns include:")
    print("     • ᜐᜒᜇ → seda, sera, sira (Combined ambiguity)")
    print("     • ᜆᜓᜇᜓ → todo, toro, turo (Combined ambiguity)")
    print("     • ᜐᜓᜎᜓ → Sulu, solo, sulo (O/U ambiguity)")
    print()
    print("  '4 unique words' patterns include:")
    print("     • ᜇᜓᜂᜈ᜔ → doon, duon, roon, ruon (Combined ambiguity)")
    print()
    
    print("💡 KEY INSIGHT:")
    print("   Most ambiguous patterns (2 unique words) represent simple")
    print("   binary choices for your disambiguation model.")
    print("   Patterns with 3+ words are more complex multi-way ambiguities.")

if __name__ == "__main__":
    analyze_group_sizes()