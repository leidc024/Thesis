# Context-Aware Baybayin Transliteration Disambiguation

A graph-based approach for disambiguating Baybayin OCR transliterations using contextual embeddings and linguistic features.

## 🎯 Results

| Method | Ambiguous Word Accuracy |
|--------|------------------------|
| MaBaybay Default | ~38% |
| Embedding-Only (WE-Only) | ~66% |
| bAI-bAI WE-Only (reported) | 77.46% |
| **bAI-bAI LLM (reported)** | **90.52%** |
| **Our Graph+Features** | **92.72%** ✓ |

**Key Achievement:** Our method exceeds the LLM-based approach (+2.2 percentage points) while being significantly faster (no API calls required).

## 📊 Test Configuration

- **Test Set:** 500 sentences (balanced by ambiguity type)
- **Total Words:** 4,500
- **Ambiguous Words:** 756
- **Clean Evaluation:** Test sentences excluded from corpus statistics

## 🏗️ Architecture

```
Input: OCR Candidates → [word1, [cand_a, cand_b], word3, ...]
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                        │
├─────────────────────────────────────────────────────────────┤
│  1. Semantic Similarity (RoBERTa)         Weight: 0.3       │
│     - Filipino RoBERTa: jcblaise/roberta-tagalog-base       │
│     - Contextual embeddings for candidate vs. sentence      │
│                                                              │
│  2. Corpus Frequency                      Weight: 0.4       │
│     - ~286k words from Filipino text corpora                │
│     - Log-normalized frequency scores                       │
│                                                              │
│  3. Co-occurrence (Bigrams)               Weight: 0.2       │
│     - P(word | prev_word) + P(next_word | word)            │
│     - Laplace smoothing for unseen bigrams                  │
│                                                              │
│  4. Morphological Features                Weight: 0.1       │
│     - Filipino prefix/suffix patterns                       │
│     - Reduplication detection                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
           Combined Score = Σ(weight_i × feature_i)
                              ↓
              Select candidate with highest score
                              ↓
Output: Disambiguated sentence → [word1, cand_a, word3, ...]
```

## 🔤 Ambiguity Types Handled

| Baybayin | Latin Options | Example |
|----------|---------------|---------|
| ᜁ | E / I | "sila" vs "sela" |
| ᜂ | O / U | "buto" vs "boto" |
| ᜇ | D / R | "dito" vs "rito" |

## 📁 Project Structure

```
Thesis/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── disambiguator.py          # Main disambiguation model
│   ├── corpus.py                 # Corpus statistics module
│   ├── morphology.py             # Filipino morphology analyzer
│   └── baselines.py              # Baseline models for comparison
│
├── evaluate.py                   # Main evaluation script
│
├── dataset/
│   ├── processed/
│   │   ├── test_sentences_500.txt    # Test sentences
│   │   └── candidates_results_v2.json # OCR candidates
│   └── results/
│       └── evaluation_results.json   # Evaluation results
│
├── Tagalog_Literary_Text.txt     # Literary corpus (~200k words)
├── Tagalog_Religious_Text.txt    # Religious corpus (~90k words)
│
└── MaBaybay-OCR/                 # OCR system
    └── Filipino Word Corpus/
        └── Tagalog_words_74419+.csv
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch transformers scikit-learn numpy tqdm networkx
```

### Run Evaluation

```bash
# Basic evaluation
python evaluate.py

# With baseline comparisons
python evaluate.py --baselines

# Custom weights
python evaluate.py --weights 0.3 0.4 0.2 0.1
```

### Use in Code

```python
from src import BaybayinDisambiguator

# Initialize
model = BaybayinDisambiguator(
    corpus_files=["Tagalog_Literary_Text.txt", "Tagalog_Religious_Text.txt"]
)

# Disambiguate
candidates = ["ang", ["dito", "rito"], "ay", "maganda"]
result, debug = model.disambiguate(candidates)
print(result)  # ['ang', 'dito', 'ay', 'maganda']
```

## 📈 Methodology

### 1. Data Preparation
- Extracted 500 balanced test sentences
- Generated OCR candidates using MaBaybay OCR simulator
- Distribution: E/I (37.5%), O/U (37.5%), D/R (15%), Combined (10%)

### 2. Feature Engineering
- **Semantic:** RoBERTa embeddings capture contextual meaning
- **Frequency:** Common words in Filipino corpora are favored
- **Co-occurrence:** Bigram statistics model word sequences
- **Morphology:** Filipino affix patterns validate word structure

### 3. Clean Evaluation
- Test sentences excluded from corpus statistics
- Prevents data leakage between train/test
- Results are unbiased and generalizable

## 📚 References

- **RoBERTa Tagalog:** Cruz & Cheng (2020) - `jcblaise/roberta-tagalog-base`
- **bAI-bAI Paper:** Baseline comparison for WE-Only and LLM approaches
- **MaBaybay OCR:** Baybayin character recognition system

## 📝 Citation

```bibtex
@thesis{baybayin_disambiguation_2024,
  title={Context-Aware Baybayin Transliteration Disambiguation},
  author={[Your Name]},
  year={2024}
}
```

## 📄 License

MIT License
