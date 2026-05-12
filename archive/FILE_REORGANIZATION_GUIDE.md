# File Reorganization Guide: MLM-PLL & Statistics Focus

## Current Directory Structure → Recommended Structure

### **KEEP (Core Files)**

```
thesis/
├── src/
│   ├── __init__.py
│   ├── disambiguator.py          ✓ KEEP (use MLM-PLL only)
│   └── corpus.py                  ✓ KEEP (corpus statistics)
├── corpus/
│   ├── Tagalog_Literary_Text.txt  ✓ KEEP
│   ├── Tagalog_Religious_Text.txt ✓ KEEP
│   └── Tagalog_Balita_Texts_Balanced.txt ✓ KEEP
├── disambiguate.py                ✓ KEEP (MATLAB wrapper)
├── evaluate.py                    ✓ KEEP (evaluation framework)
└── gold_standard_dataset/
    ├── sentences/                 ✓ KEEP (for testing)
    └── results/                   ✓ KEEP (evaluation outputs)
```

### **REMOVE/ARCHIVE (Not Currently Used)**

```
# Option 1: DELETE
src/morphology.py                 ✗ DELETE (not used with MLM-PLL alone)
src/baselines.py                  ✗ DELETE (embedding-only, LLM baselines not used)

# Option 2: ARCHIVE to old_scripts/
tests/                            → archive/old_tests/
image_generation/                 → archive/old_image_generation/
CHAPTER_3_3_IMPLEMENTATION.md     → archive/ (if superseded)
WORKFLOW.txt                      → archive/ (if superseded)
```

---

## Recommended Changes to Keep Files

### **1. Simplify `src/disambiguator.py`**

**Current:** Uses multi-feature weights and morphology
**Keep:** Only MLM-PLL scoring core

```python
# Change this:
DEFAULT_WEIGHTS = {
    'semantic': 1.0,
    'frequency': 0.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}

# To remove the option entirely - just use MLM-PLL directly
# Remove all references to: frequency, cooccurrence, morphology
```

**Lines to simplify:**
- Remove `score_candidate()` method (only needed for multi-feature)
- Keep `_get_candidate_pll()` (core PLL computation)
- Keep `_get_mlm_scores()` (MLM normalization)
- Remove weight system entirely
- Simplify `disambiguate()` to only use MLM-PLL

---

### **2. Keep `src/corpus.py` As-Is**

This is good for:
- Loading and caching corpus statistics
- Computing word frequency scores (useful fallback)
- Efficient data loading with exclusion mechanism

**No changes needed** - it's lightweight and self-contained.

---

### **3. Simplify `disambiguate.py`**

**Current:** Handles complex weight logic
**Simplify:** Remove all weight handling

```python
# Remove this section:
if num_words == 1 or num_ambiguous == num_words:
    weights = {
        'semantic': 0.0,
        'frequency': 0.1,
        'cooccurrence': 0.0,
        'morphology': 0.0
    }
else:
    weights = None

# Just initialize directly:
model = BaybayinDisambiguator(
    corpus_files=[...],
    weights=None  # Remove this param, only use MLM-PLL
)
```

---

### **4. Simplify `evaluate.py`**

**Keep:** Core evaluation framework
**Remove:** Baseline comparisons (embedding-only, LLM, MaBaybay default)

**What to remove:**
- `--baselines` argument
- `--llm` argument
- `--weights` argument
- All baseline model comparisons
- Comparison table with bAI-bAI

**What to keep:**
- Test data loading
- MLM-PLL evaluation
- Metrics computation
- Result saving

**Simplified evaluate.py:**
```python
def main():
    # ... load test data ...
    
    # Initialize model (MLM-PLL only)
    model = BaybayinDisambiguator(
        corpus_files=CORPUS_FILES,
        exclude_sentences=test_sentences
    )
    
    # Evaluate (no weights argument)
    metrics, results = model.evaluate(test_data)
    
    # Print results
    print(f"Ambiguous Accuracy: {metrics['ambiguous_accuracy']:.2%}")
    
    # Save results
    # ... save to JSON ...
```

---

## File Reorganization Steps

### **Step 1: Archive Unused Files**

```bash
# Create archive directories
mkdir -p archive/old_tests
mkdir -p archive/old_baselines
mkdir -p archive/old_scripts

# Move unused files
mv src/morphology.py archive/old_baselines/
mv src/baselines.py archive/old_baselines/
mv tests/ archive/old_tests/
mv image_generation/ archive/old_image_generation/
```

### **Step 2: Update `src/__init__.py`**

**Before:**
```python
from .disambiguator import BaybayinDisambiguator
from .corpus import CorpusStatistics
from .morphology import MorphologicalAnalyzer
from .baselines import MaBaybayDefault, EmbeddingOnly, LLMBaseline
```

**After:**
```python
from .disambiguator import BaybayinDisambiguator
from .corpus import CorpusStatistics
```

### **Step 3: Clean Up Imports in `disambiguator.py`**

**Remove:**
```python
from .morphology import MorphologicalAnalyzer
from sklearn.metrics.pairwise import cosine_similarity  # Only used for embeddings
```

**Keep:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .corpus import CorpusStatistics
```

---

## Resulting Clean File Structure

```
thesis/
├── src/
│   ├── __init__.py                    # Imports: disambiguator, corpus
│   ├── disambiguator.py               # MLM-PLL core + integration
│   └── corpus.py                      # Corpus statistics
├── corpus/                             # Text corpora
│   ├── Tagalog_Literary_Text.txt
│   ├── Tagalog_Religious_Text.txt
│   └── Tagalog_Balita_Texts_Balanced.txt
├── gold_standard_dataset/             # Test data
│   ├── sentences/                     # Test sentences
│   └── results/                       # Evaluation results
├── disambiguate.py                    # MATLAB wrapper (simplified)
├── evaluate.py                        # Evaluation script (simplified)
├── APPENDIX_CODE_SNIPPETS.md          # Updated appendix (MLM-PLL focus)
│
├── archive/                           # Archived (not used now)
│   ├── old_tests/
│   ├── old_baselines/
│   ├── old_image_generation/
│   └── old_scripts/
│
└── [Data files, venv, .git, etc.]
```

---

## Summary: What Changes

| Component | Before | After | Why |
|-----------|--------|-------|-----|
| **Disambiguator** | Multi-feature + MLM | MLM-PLL only | Focus on proven approach |
| **Baselines** | 3 baselines (default, WE-only, LLM) | None | Not using for evaluation |
| **Morphology** | Active feature | Archived | Not needed for MLM-PLL |
| **Corpus** | Supports multi-feature | As-is | Still used for frequency fallback |
| **Evaluate** | Tests multiple methods | Single MLM-PLL test | Clean, focused evaluation |
| **Weights** | Complex system | Removed | Only MLM-PLL, no weighting |

---

## Benefits of This Reorganization

✅ **Simpler codebase** - Easier to understand and maintain  
✅ **Faster startup** - No morphology analyzer to load  
✅ **Clear focus** - Only MLM-PLL + corpus statistics  
✅ **Cleaner evaluation** - No confusing baseline comparisons  
✅ **Easier debugging** - Fewer moving parts to track  
✅ **Reduced dependencies** - No need for morphology patterns  
✅ **Preserved history** - Old code in archive/ if needed later  

---

## Implementation Order

1. **First:** Move unused files to archive/
2. **Then:** Update imports in src/__init__.py
3. **Next:** Simplify disambiguator.py (remove morphology, weights)
4. **Then:** Simplify disambiguate.py (remove weight logic)
5. **Finally:** Simplify evaluate.py (remove baselines)

**Estimated time:** ~30 minutes for all changes
