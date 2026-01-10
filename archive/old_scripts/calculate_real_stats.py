"""
Calculate correct statistics for genuine ambiguous patterns
Excludes false positives (same word repeated)
"""
import csv
from collections import defaultdict

def calculate_real_statistics():
    # Categories we care about
    genuine_categories = ['E/I', 'O/U', 'D/R', 'COMBINED']
    
    stats = {
        'total_patterns': 0,
        'total_words': 0,
        'by_category': {},
        'pattern_details': []
    }
    
    # Initialize category counts
    for cat in genuine_categories:
        stats['by_category'][cat] = {'patterns': 0, 'words': 0}
    
    # Also track real UNKNOWN patterns (different words, not duplicates)
    stats['by_category']['REAL_UNKNOWN'] = {'patterns': 0, 'words': 0}
    
    with open('dataset/analysis/ambiguous_pairs_complete.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 4:
                continue
                
            baybayin = row[0]
            ambiguity_type = row[1]
            word_count = int(row[2])
            words_str = row[3].strip('"')
            words = [w.strip() for w in words_str.split(',')]
            unique_words = list(set(words))
            
            # Check if this is a genuine ambiguity (different words)
            if len(unique_words) > 1:
                if ambiguity_type in genuine_categories:
                    stats['by_category'][ambiguity_type]['patterns'] += 1
                    stats['by_category'][ambiguity_type]['words'] += len(unique_words)
                    stats['total_patterns'] += 1
                    stats['total_words'] += len(unique_words)
                    
                    stats['pattern_details'].append({
                        'baybayin': baybayin,
                        'type': ambiguity_type,
                        'unique_words': unique_words,
                        'total_instances': word_count
                    })
                
                elif ambiguity_type == 'UNKNOWN':
                    stats['by_category']['REAL_UNKNOWN']['patterns'] += 1
                    stats['by_category']['REAL_UNKNOWN']['words'] += len(unique_words)
    
    return stats

def display_statistics():
    stats = calculate_real_statistics()
    
    print("=" * 70)
    print("CORRECTED DATASET STATISTICS")
    print("Genuine Ambiguous Patterns Only (Excludes Duplicates)")
    print("=" * 70)
    print()
    
    print("📊 MAIN STATISTICS:")
    print(f"   Discovered {stats['total_patterns']} ambiguous Baybayin patterns affecting {stats['total_words']} words.")
    print()
    
    print("📋 BREAKDOWN BY CATEGORY:")
    for category, data in stats['by_category'].items():
        if data['patterns'] > 0:
            print(f"   • {category}: {data['patterns']} patterns, {data['words']} words")
    
    print()
    print("🎯 FOR YOUR UPDATE:")
    print(f"   \"Discovered {stats['total_patterns']} ambiguous Baybayin patterns affecting {stats['total_words']} words.\"")
    print()
    
    print("📈 DETAILED BREAKDOWN:")
    print(f"   • E/I ambiguities: {stats['by_category']['E/I']['patterns']} patterns")
    print(f"   • O/U ambiguities: {stats['by_category']['O/U']['patterns']} patterns")
    print(f"   • D/R ambiguities: {stats['by_category']['D/R']['patterns']} patterns") 
    print(f"   • Combined ambiguities: {stats['by_category']['COMBINED']['patterns']} patterns")
    
    if stats['by_category']['REAL_UNKNOWN']['patterns'] > 0:
        print(f"   • Real unknown ambiguities: {stats['by_category']['REAL_UNKNOWN']['patterns']} patterns")
    
    print()
    print("🔢 SUMMARY FOR DOCUMENTATION:")
    core_patterns = (stats['by_category']['E/I']['patterns'] + 
                    stats['by_category']['O/U']['patterns'] + 
                    stats['by_category']['D/R']['patterns'] + 
                    stats['by_category']['COMBINED']['patterns'])
    
    core_words = (stats['by_category']['E/I']['words'] + 
                 stats['by_category']['O/U']['words'] + 
                 stats['by_category']['D/R']['words'] + 
                 stats['by_category']['COMBINED']['words'])
    
    print(f"   Core ambiguities (E/I, O/U, D/R, COMBINED): {core_patterns} patterns, {core_words} words")
    print(f"   Additional edge cases: {stats['by_category']['REAL_UNKNOWN']['patterns']} patterns")
    print(f"   Total genuine ambiguities: {stats['total_patterns']} patterns, {stats['total_words']} words")

if __name__ == "__main__":
    display_statistics()