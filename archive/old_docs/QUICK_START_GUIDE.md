# Quick Start Guide: Dataset Creation
## Building Your Gold-Standard Baybayin Evaluation Dataset

**Last Updated:** December 2, 2025

---

## Overview

This guide walks you through the systematic process of creating a scientifically rigorous evaluation dataset for your Baybayin disambiguation research. Follow these steps sequentially.

---

## Prerequisites

✅ You already have:
- Filipino word corpus (74,419+ words)
- Literary text corpus
- Religious text corpus
- Python environment with necessary libraries

📦 Required Python libraries:
```bash
pip install pandas numpy
```

---

## Step-by-Step Execution

### Step 0: Setup (5 minutes)

Create the directory structure:

```bash
cd c:\Users\leian\Documents\Thesis
python scripts\00_setup_directories.py
```

**Expected Output:**
- Creates `dataset/` folder with subdirectories
- Creates README.md with documentation
- Shows confirmation of directory creation

**Verify:** Check that `dataset/` folder exists with subfolders: `raw/`, `processed/`, `splits/`, `analysis/`, `documentation/`

---

### Step 1: Discover Ambiguous Pairs (10-15 minutes)

Find ALL ambiguous word pairs in your corpus:

```bash
python scripts\01_find_ambiguous_pairs.py
```

**What it does:**
- Analyzes 74,419+ words
- Converts each to Baybayin
- Groups words with identical Baybayin representation
- Classifies by ambiguity type (E/I, O/U, D/R, COMBINED)
- Generates comprehensive statistics

**Expected Output Files:**
- `dataset/analysis/ambiguous_pairs_complete.csv` - All pairs in spreadsheet format
- `dataset/analysis/ambiguous_pairs_complete.json` - Detailed JSON with metadata
- `dataset/analysis/ambiguity_statistics.txt` - Summary statistics

**Review This:**
Open `ambiguous_pairs_complete.csv` and examine:
- How many E/I pairs exist?
- How many O/U pairs exist?
- What are the largest ambiguous groups?
- Are there unexpected patterns?

**Estimated Results:**
- ~500-1,500 ambiguous patterns
- ~2,000-5,000 ambiguous words total
- E/I and O/U should dominate
- Some interesting multi-way ambiguities (3+ words mapping to same Baybayin)

---

### Step 2: Extract Sentences from Corpora (10-20 minutes)

Mine sentences containing ambiguous words:

```bash
python scripts\02_extract_sentences.py
```

**What it does:**
- Reads Literary and Religious text files
- Segments into sentences
- Identifies sentences with ambiguous words
- Classifies by density (low/medium/high)
- Estimates context difficulty (easy/medium/hard)
- Filters by length (5-20 words)

**Expected Output Files:**
- `dataset/raw/candidate_sentences.json` - All extracted candidates with metadata

**Review This:**
Open `candidate_sentences.json` and check:
- Total number of candidates extracted
- Distribution by ambiguity type
- Distribution by difficulty level
- Example sentences from each category

**Estimated Results:**
- 500-2,000 candidate sentences (depends on corpus size)
- May be imbalanced toward certain types
- Some categories may be underrepresented

**Action Required:**
Based on the extraction results, identify gaps:
- Which ambiguity types are underrepresented?
- Which difficulty levels need more examples?
- Which ambiguous pairs have zero examples?

Take notes for the next step!

---

### Step 3: Balance and Curate Dataset (1-2 hours)

**Manual Step - Strategic Curation**

Based on Step 2 results, you need to:

1. **Review candidates** in `candidate_sentences.json`
2. **Select sentences** to meet distribution targets:
   - E/I: 35% (175-350 sentences for 500-1000 total)
   - O/U: 35%
   - D/R: 15%
   - Combined: 10%
   - Control: 5%

3. **Fill gaps** by constructing new sentences for:
   - Underrepresented ambiguous pairs
   - Specific difficulty levels
   - High-density scenarios

4. **Create final sentence file**:
   - One sentence per line
   - Save as `dataset/processed/filipino_sentences_v1.txt`

**Construction Guidelines for Gap-Filling:**

When creating sentences manually:
- ✅ Use natural Filipino sentence patterns
- ✅ Ensure grammatical correctness
- ✅ Include strong/weak context deliberately (for difficulty levels)
- ✅ Keep 5-20 words length
- ✅ Use only words from the 74,419+ corpus
- ✅ Make sentences culturally appropriate

**Example Constructions:**

```
# E/I - Easy context (bote vs buti)
May nakita akong bote ng tubig sa mesa.
(I saw a bottle of water on the table)
→ Strong context: "bottle" is obvious with "water"

# E/I - Hard context (bote vs buti)
Ang bote ay nasa bahay.
(The bottle/good is at home)
→ Weak context: both interpretations plausible

# O/U - Medium context (boto vs buto)
Bumili ako ng mangga na walang buto.
(I bought a mango without seeds)
→ Moderate context: "mango" suggests "seeds"

# Combined - High density
Ituro mo kung paano magluto ng bote para sa boto.
(Teach how to cook bottle/good for vote/bone)
→ Multiple ambiguities in one sentence
```

**Balancing Strategy:**

Use stratified sampling:
1. Count available candidates per category
2. Calculate selection ratios to meet targets
3. Prioritize high-quality, natural sentences
4. Construct sentences only for gaps
5. Document source (extracted vs constructed)

**Tool Recommendation:**
Create a simple spreadsheet to track:
- Sentence | Source | Ambiguity Type | Density | Difficulty | Include?

---

### Step 4: Generate Baybayin Images (30 minutes)

Use your existing script with the new dataset:

```bash
# Update create_dataset.py to use new input file
# Change: SENTENCES_FILE = "filipino_sentences.txt"
# To: SENTENCES_FILE = "dataset/processed/filipino_sentences_v1.txt"

python create_dataset.py
```

**Expected Output:**
- PNG images in `baybayin_dataset_images/`
- `annotations.csv` with ground truth mappings
- High-quality images suitable for OCR

---

### Step 5: Generate OCR Candidates (5 minutes)

Create the candidate results JSON:

```bash
# Update generate_candidate_results.py paths if needed
python generate_candidate_results.py
```

**Expected Output:**
- `candidates_results.json` with all candidate words
- Validation that candidates exist in corpus

---

### Step 6: Create Train/Val/Test Splits (10 minutes)

**Script to create:** `scripts/04_create_splits.py`

Split the dataset:
- 70% training (350-700 sentences)
- 15% validation (75-150 sentences)
- 15% test (75-150 sentences)

Ensure stratification by:
- Ambiguity type
- Difficulty level
- Density level

```python
# Pseudocode for splitting
from sklearn.model_selection import train_test_split

# Load full dataset
# Create stratification key: f"{ambiguity_type}_{difficulty}"
# Split: train_val, test = train_test_split(data, test_size=0.15, stratify=keys)
# Split: train, val = train_test_split(train_val, test_size=0.176, stratify=keys)
# Save to dataset/splits/
```

---

### Step 7: Validation and Quality Checks (15 minutes)

Run comprehensive validation:

```bash
python scripts\05_validate_dataset.py
```

**Checks to implement:**
- ✅ All words exist in corpus
- ✅ No duplicate sentences
- ✅ Distribution matches targets (±5%)
- ✅ All ambiguous pairs covered (min 3 examples)
- ✅ Sentence length within bounds
- ✅ Balanced splits (similar distributions)

**Expected Output:**
- `dataset/analysis/validation_report.json`
- Pass/fail for each quality metric
- Detailed statistics

---

## Dataset Targets Summary

| Metric | Target |
|--------|--------|
| **Total Sentences** | 500-1,000 |
| **E/I Ambiguities** | 35% (175-350) |
| **O/U Ambiguities** | 35% (175-350) |
| **D/R Ambiguities** | 15% (75-150) |
| **Combined** | 10% (50-100) |
| **Control (no ambiguity)** | 5% (25-50) |
| **Low Density** | 40% (1 ambiguous word) |
| **Medium Density** | 40% (2-3 ambiguous words) |
| **High Density** | 20% (4+ ambiguous words) |
| **Easy Context** | 30% |
| **Medium Context** | 50% |
| **Hard Context** | 20% |

---

## Troubleshooting

### Problem: Not enough sentences extracted

**Solution:**
- Lower MAX_SENTENCE_LENGTH to 25 words
- Add more text corpora (Wikipedia, news articles)
- Increase manual construction percentage
- Focus on high-frequency ambiguous pairs first

### Problem: Imbalanced distribution

**Solution:**
- Oversample underrepresented categories
- Targeted construction for specific types
- Use weighted sampling in balancing script
- Document limitations in thesis

### Problem: Context difficulty hard to assess

**Solution:**
- Develop rubric with clear examples
- Test with actual disambiguation model
- Get feedback from native speakers
- Use semantic similarity scores as proxy

### Problem: Running out of unique ambiguous pairs

**Solution:**
- Reuse pairs in different sentence contexts
- Focus on most common pairs (frequency-weighted)
- Combine with less common pairs for diversity
- Document pair frequency distribution

---

## Timeline Recap

| Week | Tasks | Estimated Time |
|------|-------|----------------|
| **Week 1** | Steps 0-2: Setup, Discovery, Extraction | 2-3 hours |
| **Week 2** | Step 3: Curation & gap analysis | 5-8 hours |
| **Week 3** | Step 3: Construction & finalization | 5-8 hours |
| **Week 4** | Steps 4-7: Images, validation, documentation | 3-5 hours |
| **Total** | | **15-24 hours** |

---

## Deliverables Checklist

By the end, you should have:

### Data Files
- [ ] `dataset/processed/filipino_sentences_v1.txt` (500-1,000 sentences)
- [ ] `dataset/processed/annotations_v1.csv` (ground truth)
- [ ] `dataset/processed/candidates_results_v1.json` (OCR candidates)
- [ ] `dataset/splits/train.json` (70%)
- [ ] `dataset/splits/validation.json` (15%)
- [ ] `dataset/splits/test.json` (15%)

### Analysis Files
- [ ] `dataset/analysis/ambiguous_pairs_complete.csv` (all pairs)
- [ ] `dataset/analysis/ambiguity_statistics.txt` (discovery stats)
- [ ] `dataset/analysis/dataset_statistics.json` (final stats)
- [ ] `dataset/analysis/validation_report.json` (quality metrics)

### Documentation
- [ ] `dataset/documentation/ambiguous_pairs_reference.md`
- [ ] `dataset/documentation/construction_notes.md`
- [ ] `dataset/documentation/source_attribution.txt`

### Images
- [ ] `baybayin_dataset_images/*.png` (one per sentence)

### Thesis Integration
- [ ] Methodology section: Dataset construction process
- [ ] Results section: Dataset statistics and characteristics
- [ ] Appendix: Complete ambiguous pairs list
- [ ] Appendix: Example sentences from each category

---

## Next Steps After Dataset Creation

Once your dataset is complete:

1. **Implement Graph-Based Model**
   - Build graph representation of candidate words
   - Integrate RoBERTa Tagalog embeddings
   - Implement Personalized PageRank

2. **Baseline Comparisons**
   - Implement simple word embedding method (bAI-bAI's 77% baseline)
   - Implement frequency-based method
   - Document baseline performance

3. **Evaluation**
   - Run all methods on test set
   - Compare accuracy, speed, F1-scores
   - Analyze by ambiguity type and difficulty

4. **Thesis Writing**
   - Dataset construction methodology
   - Model architecture and reasoning
   - Results and analysis
   - Discussion of performance gap bridging

---

## Resources

### Python Libraries Needed
```bash
pip install pandas numpy scikit-learn
```

### Tagalog NLP Resources
- Filipino word corpus: Already in workspace ✅
- RoBERTa Tagalog: `jcblaise/roberta-tagalog-base` (Hugging Face)
- NetworkX: For graph algorithms

### References
- bAI-bAI study: For baseline comparison
- MaBaybay-OCR: Character recognition
- Your thesis plan document: DATASET_CREATION_PLAN.md

---

## Questions or Issues?

Common questions:

**Q: What if I can't find 1,000 unique sentences?**
A: Start with 500 as minimum viable dataset. Quality > quantity. Ensure good coverage and balance.

**Q: How do I validate constructed sentences?**
A: Check grammar with native speakers, ensure words exist in corpus, verify natural flow.

**Q: What if some ambiguous pairs have no natural sentences?**
A: Construct examples deliberately. Document in thesis as limitation or controlled construction.

**Q: How to handle dialectal variations?**
A: Focus on standard Filipino/Tagalog. Note dialectal issues in thesis discussion.

---

**Ready to begin? Start with Step 0! 🚀**
