# Gold-Standard Evaluation Dataset Creation Plan
## Context-Aware Baybayin Transliteration Disambiguation

**Date:** December 2, 2025  
**Project Phase:** Methodology - Dataset Construction

---

## 1. RESEARCH OBJECTIVES

### Primary Goal
Create a scientifically rigorous, balanced evaluation dataset for testing the graph-based Baybayin disambiguation model against existing methods (word embeddings at 77% accuracy, LLM-based methods at higher accuracy but impractical speed).

### Success Criteria
- **Comprehensive coverage** of known Baybayin ambiguities
- **Balanced distribution** of ambiguous cases
- **Sufficient size** for statistical significance
- **Representative** of real-world Filipino text
- **Reproducible** methodology for future research

---

## 2. KNOWN BAYBAYIN AMBIGUITIES

### Core Vowel Ambiguities (E/I and O/U)

Based on the Baybayin script's inherent limitations:

| Baybayin Character | Possible Latin Vowels | Reason |
|-------------------|----------------------|---------|
| ᜁ | e, i | Single character for both sounds |
| ᜂ | o, u | Single character for both sounds |

### Consonant Ambiguities

| Baybayin Character | Possible Latin Consonants | Reason |
|-------------------|--------------------------|---------|
| ᜇ | d, r | Historical orthographic merger |

### Example Ambiguous Word Pairs

**E/I Confusion:**
- bote (bottle) ↔ buti (good)
- pera (money) ↔ pira (how many)
- mesa (table) ↔ misa (mass/church service)
- lente (lens) ↔ linti (lightning)
- gera (war) ↔ gira (grind)
- sero (zero) ↔ siro (syrup)
- tela (cloth) ↔ tila (seems)
- pera (money) ↔ pira (how many)
- tubo (profit/pipe) ↔ tibo (lesbian)
- tela (fabric) ↔ tila (seemingly)

**O/U Confusion:**
- boto (vote) ↔ buto (bone/seed)
- todo (all/total) ↔ tudo (drill/teach)
- koro (choir) ↔ kuro (opinion)
- toro (bull) ↔ turo (point/teach)
- loko (crazy) ↔ luko (hollow)
- solo (solo) ↔ sulo (torch)
- puso (heart) ↔ puso (salt)
- kubo (hut) ↔ kobo (cent coin)
- lobo (balloon) ↔ lubo (complete)
- suso (snail/breast) ↔ soso (bland)

**D/R Confusion:**
- daan (hundred/path) ↔ raan (hundred - alternate)
- dami (amount) ↔ rami (abundance)
- dito (here) ↔ rito (here - contracted form)
- diyan (there) ↔ riyan (there - contracted form)

**Multiple Confusion (Combined):**
- ituro (teach) ↔ itodo (do completely) ↔ itoro (teach - less common)
- kong (my - contracted) ↔ kung (if)
- higante (giant) ↔ higanti (revenge)

---

## 3. DATASET SIZE SPECIFICATION

### Target Numbers

**Total Sentences:** 500-1,000 sentences

**Rationale:**
- Sufficient for statistical significance (>100 examples per major category)
- Comparable to similar NLP benchmark datasets
- Manageable for manual annotation/validation
- Allows for train/validation/test split (70/15/15 or 80/10/10)

### Distribution Strategy

#### By Ambiguity Type
- **E/I ambiguities:** 35% (175-350 sentences)
- **O/U ambiguities:** 35% (175-350 sentences)
- **D/R ambiguities:** 15% (75-150 sentences)
- **Combined ambiguities:** 10% (50-100 sentences)
- **Control (no ambiguity):** 5% (25-50 sentences)

#### By Ambiguity Density
- **Low density:** 1 ambiguous word per sentence (40%)
- **Medium density:** 2-3 ambiguous words per sentence (40%)
- **High density:** 4+ ambiguous words per sentence (20%)

#### By Context Difficulty
- **Easy:** Strong contextual clues make correct choice obvious (30%)
- **Medium:** Moderate context, requires semantic understanding (50%)
- **Hard:** Weak context, both candidates plausible (20%)

---

## 4. DATA SOURCES

### Primary Sources

#### 1. Existing Tagalog Corpora (In Workspace)
- ✅ `Tagalog_Literary_Text.txt` - Literary fiction for natural language patterns
- ✅ `Tagalog_Religious_Text.txt` - Formal/religious register
- ✅ `MaBaybay-OCR/Filipino Word Corpus/Tagalog_words_74419+.csv` - Validation dictionary

#### 2. Additional Recommended Sources
- **TLUnified Corpus** - Large-scale Tagalog text corpus (academic research)
- **WikiText-TL** - Filipino Wikipedia articles (encyclopedic style)
- **Filipino News Corpus** - Contemporary news articles (modern usage)
- **Social Media Tagalog** - Conversational/informal register (balanced approach)

### Validation Dictionary
Use the existing 74,419+ word corpus to ensure all generated candidates are valid Filipino words.

---

## 5. SENTENCE SELECTION METHODOLOGY

### Phase 1: Automated Mining (Weeks 1-2)

**Objective:** Extract candidate sentences containing target ambiguous words

**Process:**
1. Parse source corpora (Literary + Religious texts)
2. Identify sentences containing known ambiguous word pairs
3. Tag sentences by ambiguity type and density
4. Filter by length (5-20 words optimal for context)
5. Extract 2,000-3,000 candidate sentences

**Tools:**
- Python script with regex/NLP for sentence extraction
- spaCy or similar for Tagalog tokenization
- Pandas for data organization

### Phase 2: Strategic Curation (Weeks 2-3)

**Objective:** Select balanced, high-quality sentences

**Selection Criteria:**
- ✓ Contains target ambiguous words
- ✓ Context strength classification (easy/medium/hard)
- ✓ Natural sentence structure
- ✓ Grammatically correct
- ✓ Cultural appropriateness
- ✓ Length suitable for OCR (not too long)

**Balancing Strategy:**
1. Create stratified sample based on distribution targets
2. Ensure coverage of all major ambiguous pairs
3. Validate against distribution percentages
4. Fill gaps with constructed sentences if needed

### Phase 3: Manual Construction (Week 3)

**Objective:** Fill gaps with strategically designed sentences

**Construction Guidelines:**
- Target underrepresented ambiguous pairs
- Vary context difficulty levels
- Create natural-sounding sentences
- Embed multiple ambiguities for high-density category
- Use common Filipino sentence patterns

**Quality Checks:**
- Native speaker review (if available)
- Grammar validation
- Contextual coherence
- OCR suitability (avoid overly complex layouts)

### Phase 4: Validation & Annotation (Week 4)

**Objective:** Ensure dataset quality and completeness

**Validation Steps:**
1. **Linguistic Review:** Grammar, naturalness, appropriateness
2. **Balance Check:** Verify distribution targets met
3. **Ground Truth Annotation:** Confirm correct transliterations
4. **Ambiguity Verification:** Ensure candidates exist in corpus
5. **Difficulty Rating:** Classify context strength

**Annotation Format:**
```json
{
  "sentence_id": 1,
  "latin_text": "Bawat isa ay may boto sa halalan",
  "baybayin_text": "ᜊᜏᜆ᜔ ᜁᜐ ᜀᜌ᜔ ᜋᜌ᜔ ᜊᜓᜆᜓ ᜐ ᜑᜎᜎᜈ᜔",
  "ambiguous_words": [
    {
      "position": 5,
      "word": "boto",
      "candidates": ["boto", "buto"],
      "correct": "boto",
      "ambiguity_type": "O/U",
      "context_difficulty": "easy"
    }
  ],
  "source": "constructed",
  "metadata": {
    "ambiguity_density": "low",
    "sentence_length": 7,
    "domain": "civic"
  }
}
```

---

## 6. IDENTIFYING ADDITIONAL AMBIGUOUS PAIRS

### Systematic Discovery Methods

#### Method 1: Corpus Analysis
**Process:**
1. Extract all words from 74,419+ corpus
2. Generate Baybayin transliterations for each word
3. Group words by identical Baybayin sequences
4. Filter groups with 2+ members (ambiguous pairs/sets)
5. Rank by word frequency and semantic diversity

**Python Implementation:**
```python
def find_ambiguous_pairs(word_corpus):
    """
    Analyzes corpus to find all ambiguous word pairs
    that share the same Baybayin representation.
    """
    from collections import defaultdict
    
    baybayin_map = defaultdict(list)
    
    for word in word_corpus:
        baybayin = latin_to_baybayin(word)
        baybayin_map[baybayin].append(word)
    
    # Filter for ambiguous groups
    ambiguous_groups = {
        baybayin: words 
        for baybayin, words in baybayin_map.items() 
        if len(words) > 1
    }
    
    return ambiguous_groups
```

#### Method 2: Phonetic Pattern Analysis
**Focus Areas:**
- Common Filipino word patterns (CV, CVC, CVV structures)
- High-frequency affixes (nag-, mag-, -in, -an, etc.)
- Reduplication patterns that may create ambiguities

#### Method 3: Linguistic Research
**Resources:**
- Tagalog phonology literature
- Historical Baybayin usage studies
- Filipino linguistics dissertations
- Comparative analysis with similar syllabaries

### Expected Discoveries
- **Additional E/I pairs:** 50-100 new pairs
- **Additional O/U pairs:** 50-100 new pairs
- **Additional D/R pairs:** 20-30 new pairs
- **Compound ambiguities:** 30-50 cases

---

## 7. DATASET ORGANIZATION

### File Structure
```
thesis/
├── dataset/
│   ├── raw/
│   │   ├── literary_extracts.txt
│   │   ├── religious_extracts.txt
│   │   └── constructed_sentences.txt
│   ├── processed/
│   │   ├── filipino_sentences_v1.txt (500+ sentences)
│   │   ├── annotations_v1.csv
│   │   ├── candidates_results_v1.json
│   │   └── dataset_metadata.json
│   ├── splits/
│   │   ├── train.json (70%)
│   │   ├── validation.json (15%)
│   │   └── test.json (15%)
│   ├── analysis/
│   │   ├── ambiguity_distribution.csv
│   │   ├── word_frequency_analysis.csv
│   │   └── difficulty_ratings.csv
│   └── documentation/
│       ├── ambiguous_pairs_complete.csv
│       ├── source_attribution.txt
│       └── annotation_guidelines.md
├── baybayin_dataset_images/
└── scripts/
    ├── 01_extract_sentences.py
    ├── 02_find_ambiguous_pairs.py
    ├── 03_balance_dataset.py
    ├── 04_generate_images.py
    └── 05_validate_dataset.py
```

### Metadata Tracking
Each sentence should include:
- **Source:** (literary/religious/constructed/other)
- **Date added:** ISO format timestamp
- **Ambiguity type(s):** E/I, O/U, D/R, combined
- **Density level:** low/medium/high
- **Difficulty:** easy/medium/hard
- **Validator:** Annotator ID/name
- **Quality score:** 1-5 rating

---

## 8. QUALITY ASSURANCE METRICS

### Coverage Metrics
- ✓ All known ambiguous pairs represented (minimum 3 examples each)
- ✓ Distribution targets met within ±5%
- ✓ All difficulty levels adequately represented
- ✓ Multiple sentence lengths covered (5-20 words)

### Validity Metrics
- ✓ 100% of words exist in Filipino corpus
- ✓ 95%+ grammatical correctness (native speaker validation)
- ✓ 90%+ natural language flow
- ✓ 0% duplicate sentences

### Reliability Metrics
- ✓ Inter-annotator agreement >90% (if multiple annotators)
- ✓ Ground truth consistency check
- ✓ OCR image quality >95% recognition rate

### Diversity Metrics
- ✓ Vocabulary diversity (type-token ratio >0.4)
- ✓ Domain diversity (civic, religious, literary, everyday, etc.)
- ✓ Syntactic diversity (various sentence structures)

---

## 9. IMPLEMENTATION TIMELINE

### Week 1: Setup & Discovery
- [ ] Set up directory structure
- [ ] Implement ambiguous pair discovery script
- [ ] Run analysis on 74,419+ word corpus
- [ ] Generate comprehensive ambiguous pairs list
- [ ] Document all findings

### Week 2: Sentence Mining
- [ ] Implement sentence extraction script
- [ ] Parse literary and religious corpora
- [ ] Extract 2,000+ candidate sentences
- [ ] Initial filtering and classification
- [ ] Generate candidate pool report

### Week 3: Curation & Construction
- [ ] Strategic selection for balanced distribution
- [ ] Construct sentences for underrepresented categories
- [ ] Validate sentence quality
- [ ] Reach 500-1,000 sentence target
- [ ] Create initial annotations

### Week 4: Validation & Finalization
- [ ] Quality assurance checks
- [ ] Balance verification
- [ ] Generate dataset statistics
- [ ] Create train/validation/test splits
- [ ] Finalize documentation
- [ ] Generate Baybayin images
- [ ] Run OCR candidate generation

---

## 10. SCRIPT SPECIFICATIONS

### Script 1: Ambiguous Pair Finder
**Purpose:** Discover all ambiguous word pairs in corpus
**Input:** Tagalog_words_74419+.csv
**Output:** ambiguous_pairs_complete.csv
**Key Features:**
- Baybayin transliteration for all words
- Group by identical Baybayin representation
- Frequency analysis
- Semantic category tagging

### Script 2: Sentence Extractor
**Purpose:** Mine sentences from text corpora
**Input:** Tagalog_Literary_Text.txt, Tagalog_Religious_Text.txt
**Output:** candidate_sentences.json
**Key Features:**
- Sentence segmentation
- Ambiguous word detection
- Length filtering (5-20 words)
- Context difficulty estimation
- Metadata extraction

### Script 3: Dataset Balancer
**Purpose:** Create balanced final dataset
**Input:** candidate_sentences.json, distribution_targets.json
**Output:** filipino_sentences_v1.txt, dataset_statistics.json
**Key Features:**
- Stratified sampling
- Distribution verification
- Gap identification
- Quality scoring

### Script 4: Validator
**Purpose:** Ensure dataset quality
**Input:** filipino_sentences_v1.txt, annotations_v1.csv
**Output:** validation_report.json
**Key Features:**
- Grammar checking
- Corpus validation (all words exist)
- Duplicate detection
- Balance verification
- Coverage analysis

---

## 11. RESEARCH CONTRIBUTION

### Novel Aspects
1. **First standardized benchmark** for Baybayin disambiguation
2. **Systematic coverage** of all known ambiguity types
3. **Difficulty-stratified** evaluation (easy/medium/hard)
4. **Reproducible methodology** for dataset construction
5. **Open framework** for future expansion

### Comparison with Existing Work
**bAI-bAI Study Limitations:**
- Used ad-hoc test sentences
- No standardized benchmark
- Limited ambiguity coverage
- No difficulty stratification

**Your Contribution:**
- Comprehensive, balanced dataset
- Clear annotation standards
- Multiple difficulty levels
- Reproducible construction process
- Enables fair comparison of disambiguation methods

### Future Extensions
- Multi-modal dataset (handwritten Baybayin)
- Cross-dialectal variations
- Historical text samples
- Noisy OCR scenarios
- Real-world document samples

---

## 12. EXPECTED CHALLENGES & SOLUTIONS

### Challenge 1: Corpus Size Limitations
**Issue:** Existing corpora may not contain all ambiguous pairs in sufficient quantity
**Solution:** 
- Combine multiple corpora
- Strategic sentence construction
- Collaborate with Filipino language educators
- Use online Filipino resources (with attribution)

### Challenge 2: Context Difficulty Classification
**Issue:** Subjective assessment of what makes context "easy" vs "hard"
**Solution:**
- Develop clear rubric with examples
- Multiple annotator review
- Pilot testing with disambiguation models
- Quantitative metrics (semantic similarity scores)

### Challenge 3: Native Speaker Validation
**Issue:** Access to qualified native speakers for validation
**Solution:**
- Online Filipino language communities
- University Filipino departments
- Language exchange platforms
- Self-validation with linguistic resources

### Challenge 4: Dataset Imbalance
**Issue:** Some ambiguous pairs may be rare in natural text
**Solution:**
- Constructed sentences for rare pairs
- Weighted sampling strategies
- Document limitations in thesis
- Focus on high-frequency pairs for core evaluation

---

## 13. SUCCESS METRICS

### Quantitative Goals
- ✓ **500-1,000 sentences** total
- ✓ **100+ unique ambiguous word pairs** covered
- ✓ **Distribution variance <5%** from targets
- ✓ **OCR candidate accuracy >95%** (correct candidates in list)
- ✓ **Inter-annotator agreement >90%**

### Qualitative Goals
- ✓ Natural, grammatically correct Filipino sentences
- ✓ Diverse domains and registers
- ✓ Appropriate cultural content
- ✓ Suitable for OCR processing
- ✓ Clear ground truth annotations

### Impact Goals
- ✓ Enables rigorous evaluation of graph-based model
- ✓ Supports comparison with bAI-bAI baseline (77% accuracy)
- ✓ Provides benchmark for future research
- ✓ Demonstrates clear methodology for thesis

---

## REFERENCES & RESOURCES

### Academic Resources
- Filipino linguistic literature on Baybayin ambiguities
- Tagalog phonology and orthography studies
- NLP dataset construction best practices
- Similar syllabary disambiguation research (e.g., Devanagari)

### Technical Resources
- spaCy for Tagalog (if available)
- Python NLP libraries (NLTK, scikit-learn)
- RoBERTa Tagalog model documentation
- Graph analysis libraries (NetworkX)

### Data Resources
- ✅ MaBaybay-OCR corpus (74,419+ words)
- ✅ Literary text corpus
- ✅ Religious text corpus
- TLUnified Corpus (to acquire)
- Filipino Wikipedia dumps (public domain)

---

## NEXT IMMEDIATE STEPS

1. **Create directory structure** (dataset/, scripts/, etc.)
2. **Implement ambiguous pair finder** (Script 1)
3. **Run discovery analysis** on 74,419+ corpus
4. **Generate complete ambiguous pairs list** (CSV)
5. **Document findings** in thesis methodology section

---

**Document Version:** 1.0  
**Last Updated:** December 2, 2025  
**Status:** Planning Complete - Ready for Implementation
