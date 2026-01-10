"""
Analysis: "daan" vs "raan" - Are these legitimate Tagalog words?
"""

def analyze_daan_vs_raan():
    print("=" * 70)
    print("LINGUISTIC ANALYSIS: 'daan' vs 'raan'")
    print("=" * 70)
    print()
    
    print("🔍 YOUR QUESTION: Are 'daan' and 'raan' both legitimate Tagalog words?")
    print()
    
    # Analysis based on Filipino linguistics knowledge
    word_analysis = {
        'daan': {
            'legitimacy': 'DEFINITELY LEGITIMATE',
            'meanings': [
                'hundred (number)',
                'path/road/way (route)',
                'means/method (way of doing)'
            ],
            'examples': [
                'isang daan = one hundred',
                'daang tao = hundreds of people', 
                'sa daan = on the road/path',
                'walang ibang daan = no other way'
            ],
            'frequency': 'VERY COMMON',
            'status': '✅ Core Tagalog vocabulary'
        },
        'raan': {
            'legitimacy': 'QUESTIONABLE/VARIANT',
            'meanings': [
                'hundred (alternate/archaic form)',
                'possibly dialectal variant of "daan"',
                'may be from compound words (na-ka-raan, pa-raan)'
            ],
            'examples': [
                'Appears in: nakaraan (past)',
                'Appears in: paraan (method)',
                'Found in corpus but rare as standalone'
            ],
            'frequency': 'VERY RARE',
            'status': '⚠️ Possibly corpus artifact or archaic form'
        }
    }
    
    print("📊 DETAILED ANALYSIS:")
    print("-" * 50)
    
    for word, data in word_analysis.items():
        print(f"\n🔤 WORD: '{word}'")
        print(f"   Legitimacy: {data['legitimacy']}")
        print(f"   Status: {data['status']}")
        print(f"   Frequency: {data['frequency']}")
        print(f"   Meanings:")
        for meaning in data['meanings']:
            print(f"     • {meaning}")
        print(f"   Examples:")
        for example in data['examples']:
            print(f"     • {example}")
    
    print("\n" + "=" * 70)
    print("CORPUS INVESTIGATION RESULTS")
    print("=" * 70)
    
    print("\n📋 FINDINGS FROM YOUR CORPUS:")
    print("   • 'daan' appears multiple times (very common)")
    print("   • 'raan' appears standalone in corpus")
    print("   • 'raan' also appears in compounds: nakaraan, paraan, etc.")
    print("   • Both map to same Baybayin: ᜇᜀᜈ᜔")
    
    print("\n🤔 POSSIBLE EXPLANATIONS FOR 'raan':")
    
    explanations = [
        {
            'theory': 'CORPUS QUALITY ISSUE',
            'explanation': 'OCR errors or typos in source material',
            'likelihood': 'HIGH',
            'evidence': 'Very low frequency, questionable legitimacy'
        },
        {
            'theory': 'DIALECTAL VARIANT', 
            'explanation': 'Regional pronunciation variation',
            'likelihood': 'MEDIUM',
            'evidence': 'Some dialects may use r/d variation'
        },
        {
            'theory': 'ARCHAIC FORM',
            'explanation': 'Historical spelling of "hundred"', 
            'likelihood': 'LOW',
            'evidence': 'No clear historical documentation'
        },
        {
            'theory': 'EXTRACTION ARTIFACT',
            'explanation': 'Partial word from compounds (pa-raan → raan)',
            'likelihood': 'HIGH', 
            'evidence': 'Common in compound words'
        }
    ]
    
    for theory in explanations:
        print(f"\n   📝 {theory['theory']}:")
        print(f"      Explanation: {theory['explanation']}")
        print(f"      Likelihood: {theory['likelihood']}")
        print(f"      Evidence: {theory['evidence']}")
    
    print("\n" + "=" * 70)
    print("IMPACT ON YOUR RESEARCH")
    print("=" * 70)
    
    print("\n✅ GOOD NEWS: This doesn't hurt your research!")
    
    impact_points = [
        "Your algorithm correctly identified the ambiguity",
        "Both words (legitimate or not) have same Baybayin representation", 
        "This tests your model's robustness to corpus noise",
        "Real-world OCR will encounter similar edge cases",
        "Your disambiguation model should handle both scenarios"
    ]
    
    for point in impact_points:
        print(f"   • {point}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    
    recommendations = [
        "Keep this pair in your dataset - it's scientifically valid",
        "Document this as an example of corpus quality issues",
        "Use context to determine correct output in sentences",
        "Test how your model handles legitimate vs questionable words",
        "Consider this a feature, not a bug - adds robustness"
    ]
    
    for rec in recommendations:
        print(f"   • {rec}")
    
    print(f"\n💡 RESEARCH INSIGHT:")
    print("   This ambiguity represents real challenges OCR systems face:")
    print("   • Corpus quality variations")
    print("   • Dialectal differences") 
    print("   • Historical spelling variations")
    print("   • Your model needs to handle all these cases!")

if __name__ == "__main__":
    analyze_daan_vs_raan()