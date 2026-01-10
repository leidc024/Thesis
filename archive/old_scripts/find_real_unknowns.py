"""
Find real UNKNOWN ambiguous patterns with different words
"""
import csv

def find_real_unknowns():
    real_unknowns = []
    
    with open('dataset/analysis/ambiguous_pairs_complete.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        
        for row in reader:
            if len(row) >= 4 and row[1] == 'UNKNOWN':
                words_str = row[3].strip('"')
                words = [w.strip() for w in words_str.split(',')]
                unique_words = list(set(words))
                
                if len(unique_words) > 1:  # Actually different words
                    real_unknowns.append({
                        'baybayin': row[0],
                        'count': int(row[2]),
                        'unique_words': unique_words,
                        'all_words': words
                    })
    
    return real_unknowns

if __name__ == "__main__":
    unknowns = find_real_unknowns()
    
    print("=" * 70)
    print("REAL UNKNOWN AMBIGUOUS PATTERNS")
    print("=" * 70)
    print()
    
    if unknowns:
        print(f"Found {len(unknowns)} real UNKNOWN ambiguous patterns:")
        print()
        
        for i, pattern in enumerate(unknowns[:10], 1):  # Show first 10
            print(f"{i}. Baybayin: {pattern['baybayin']}")
            print(f"   Words: {' ↔ '.join(pattern['unique_words'])}")
            print(f"   Total instances: {pattern['count']}")
            print()
    else:
        print("❌ NO REAL UNKNOWN AMBIGUOUS PATTERNS FOUND!")
        print()
        print("🔍 This means ALL 'UNKNOWN' patterns are actually:")
        print("   • The same word repeated multiple times")
        print("   • Data quality issues in your corpus")
        print("   • Not true ambiguities")
        print()
        print("💡 EXPLANATION:")
        print("   • Your corpus has duplicate entries")
        print("   • 'sama' appears 73 times → creates 'ambiguous' group")
        print("   • But it's just 1 word, not multiple different words")
        print("   • These should be filtered out as false positives")