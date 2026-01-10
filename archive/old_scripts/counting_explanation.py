"""
AMBIGUITY COUNTING EXPLANATION

This script explains how ambiguities are counted in your dataset
and clarifies the difference between individual ambiguities vs patterns.
"""

def explain_counting():
    print("=" * 70)
    print("AMBIGUITY COUNTING: How It Actually Works")
    print("=" * 70)
    print()
    
    print("🤔 YOUR QUESTION: Is 'bote' vs 'buti' counted as 3 ambiguities?")
    print("   (2 for E/I + 1 for O/U)")
    print()
    print("📊 ANSWER: NO - It's counted as 1 PATTERN with COMBINED type")
    print()
    
    print("🔍 LET'S EXAMINE THE REAL DATA:")
    print("-" * 40)
    
    # Example from your actual dataset
    examples = [
        {
            "baybayin": "ᜊᜓᜆᜒ",
            "words": ["bote", "buti"],
            "type": "COMBINED",
            "individual_ambiguities": ["o→u (bote→buto)", "e→i (bote→biti)"],
            "pattern_count": 1
        },
        {
            "baybayin": "ᜎᜃᜒ", 
            "words": ["lake", "laki"],
            "type": "E/I",
            "individual_ambiguities": ["e→i (lake→laki)"],
            "pattern_count": 1
        },
        {
            "baybayin": "ᜆᜓᜇᜓ",
            "words": ["todo", "toro", "turo"],
            "type": "COMBINED",
            "individual_ambiguities": ["o→u (todo→tudo)", "o→u (toro→turu)", "d→r (todo→rodo)"],
            "pattern_count": 1
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"{i}. PATTERN: {ex['baybayin']}")
        print(f"   Words: {' ↔ '.join(ex['words'])}")
        print(f"   Classification: {ex['type']}")
        print(f"   Contains these individual ambiguities:")
        for amb in ex['individual_ambiguities']:
            print(f"     • {amb}")
        print(f"   📊 Counted as: {ex['pattern_count']} pattern")
        print()
    
    print("=" * 70)
    print("COUNTING METHODOLOGY")
    print("=" * 70)
    print()
    
    print("🎯 PATTERN-BASED COUNTING (What your dataset uses):")
    print("   • Each unique Baybayin representation = 1 pattern")
    print("   • 'bote' ↔ 'buti' = 1 COMBINED pattern")
    print("   • 'lake' ↔ 'laki' = 1 E/I pattern")  
    print("   • 'todo' ↔ 'toro' ↔ 'turo' = 1 COMBINED pattern")
    print()
    
    print("📈 YOUR ACTUAL STATISTICS:")
    print("   • Total patterns: 10,122")
    print("   • E/I patterns: 465 (pure E/I only)")
    print("   • O/U patterns: 491 (pure O/U only)")
    print("   • D/R patterns: 229 (pure D/R only)")
    print("   • COMBINED patterns: 93 (mixed types)")
    print("   • UNKNOWN patterns: 8,844 (other ambiguities)")
    print()
    
    print("🔍 INDIVIDUAL AMBIGUITY COUNTING (Alternative view):")
    print("   If we counted each individual vowel/consonant ambiguity:")
    print("   • 'bote' ↔ 'buti' would contribute:")
    print("     - 1 count to O/U (o in position 1)")
    print("     - 1 count to E/I (e in position 3)")
    print("     - Total: 2 individual ambiguities")
    print()
    
    print("✅ WHY PATTERN COUNTING MAKES SENSE:")
    print("   • Each pattern represents 1 disambiguation challenge")
    print("   • When OCR sees 'ᜊᜓᜆᜒ', it faces 1 decision: bote or buti?")
    print("   • The fact that it involves both O/U and E/I is metadata")
    print("   • Your graph model needs to resolve 1 pattern, not count ambiguities")
    print()
    
    print("📊 DATASET DISTRIBUTION TARGET:")
    print("   Your 500-sentence dataset aims for:")
    print("   • 175 sentences with E/I patterns (35%)")
    print("   • 175 sentences with O/U patterns (35%)")  
    print("   • 75 sentences with D/R patterns (15%)")
    print("   • 50 sentences with COMBINED patterns (10%)")
    print("   • 25 sentences with no ambiguity (5%)")

if __name__ == "__main__":
    explain_counting()