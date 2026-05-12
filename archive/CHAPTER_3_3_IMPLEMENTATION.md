# Chapter 3.3: Implementation

## 3.3.1 Disambiguation Layer Integration with MaBaybay-OCR

The integration leverages a Python-based disambiguation engine (BaybayinDisambiguator), invoked through MATLAB via a system call. The workflow proceeds as follows: (1) MaBaybay-OCR produces up to three transliteration candidates per word position as a MATLAB cell array, (2) a MATLAB wrapper script converts these candidates to JSON format, (3) a Python entry point script loads these candidates and the disambiguation engine scores each candidate using either contextual (MLM-PLL) or frequency-based features, and (4) the script returns the disambiguated sentence to standard output (stdout), which MATLAB captures and processes further.

This modular design preserves the original OCR pipeline while adding a sophisticated post-processing layer without requiring modifications to the MATLAB codebase. By default, MaBaybay-OCR selects only the first candidate from its three possible transliterations. Our integration instead captures all generated candidates, allowing context-aware selection to override MaBaybay's default choice when warranted by sentence context.

### Key Components

**BaybayinDisambiguator Class (src/disambiguator.py)**

This is the core engine responsible for scoring and candidate selection. Upon initialization, it loads: (1) the RoBERTa-Tagalog pre-trained model, and (2) corpus statistics computed from three Filipino text sources (Tagalog literary texts, Tagalog religious texts, and Tagalog news texts). The disambiguate() method processes the candidate list position-by-position. For ambiguous positions (where candidates appear as a list), it computes MLM-PLL scores for each candidate and selects the one with the highest score.

**Wrapper Script (disambiguate.py)**

Complementing the core engine, a wrapper script serves as the interface between the OCR output and the Python environment. This script handles the conversion of MATLAB cell arrays to Python-compatible formats, detects context availability to assign appropriate scoring weights, and suppresses verbose model-loading logs to maintain a clean output stream for the MATLAB console. The wrapper encapsulates all MATLAB integration logic, allowing the core BaybayinDisambiguator class to remain implementation-agnostic.

---

## 3.3.2 Pseudo-Log-Likelihood (PLL) Scoring for Context-Aware Disambiguation

The semantic context feature was implemented using the Masked Language Model (MLM) head from RoBERTa, specifically through Pseudo-Log-Likelihood (PLL) scoring. This approach directly evaluates how well each candidate word fits within the sentence context by measuring the probability of the candidate under the language model.

The implementation works as follows: for each ambiguous word position, a tokenized sentence is created with that position masked. The MLM model then produces logits (unnormalized log-probabilities) for all vocabulary items at the masked position. Rather than simply selecting the highest-probability token, PLL scoring uses a more principled approach that handles multi-token words. Each subtoken of the candidate word is masked individually, and the log-probability of recovering the original token is summed across all subtokens.

### PLL Computation

The core PLL computation is implemented as follows:

```python
def _get_candidate_pll(
    self,
    sentence_words: List[str],
    position: int,
    candidate: str
) -> float:
    """
    Compute Pseudo-Log-Likelihood (PLL) score for a candidate word.
    
    PLL measures how well a word fits in context by masking each of its
    subtokens one at a time and summing the log-probabilities.
    Higher PLL = more natural/probable word in that context.
    
    Reference: Salazar et al. (2020) "Masked Language Model Scoring"
    
    Args:
        sentence_words: List of words in the sentence
        position: Index of the ambiguous word position
        candidate: Candidate word to score
        
    Returns:
        PLL score (negative float, higher = better fit)
    """
    # Build sentence with candidate inserted
    words = list(sentence_words)
    words[position] = candidate
    full_sentence = ' '.join(words)
    
    # Tokenize the full sentence
    encoding = self.tokenizer(
        full_sentence,
        return_tensors='pt',
        truncation=True,
        max_length=128
    )
    token_ids = encoding['input_ids'][0].clone().to(self.device)
    attention_mask = encoding['attention_mask'][0].to(self.device)
    
    # Find candidate's subtoken positions
    cand_tokens = self.tokenizer.encode(' ' + candidate, add_special_tokens=False)
    cand_positions = self._find_subtoken_positions(token_ids.cpu(), cand_tokens)
    
    if not cand_positions:
        # Try without space prefix (for sentence-initial words)
        cand_tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
        cand_positions = self._find_subtoken_positions(token_ids.cpu(), cand_tokens)
    
    if not cand_positions:
        return -100.0  # Very low score for unfound words
    
    # Compute PLL: mask each subtoken one at a time, sum log-probs
    total_log_prob = 0.0
    
    for pos in cand_positions:
        masked = token_ids.clone()
        original_token = masked[pos].item()
        masked[pos] = self.tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = self.model(
                masked.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0)
            )
            logits = outputs.logits[0, pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            total_log_prob += log_probs[original_token].item()
    
    return total_log_prob
```

For a candidate word with subtokens at positions $\{p_1, p_2, \ldots, p_k\}$, the PLL score is computed as:

$$\text{PLL}(\text{candidate}) = \sum_{i=1}^{k} \log P(t_i \mid \text{masked sentence at position } p_i)$$

where $t_i$ is the original token at position $p_i$.

### MLM Score Normalization

The raw PLL scores are then normalized using softmax to produce scores between 0 and 1:

```python
def _get_mlm_scores(
    self,
    sentence_words: List[str],
    position: int,
    candidates: List[str]
) -> Dict[str, float]:
    """
    Compute normalized MLM scores for all candidates at an ambiguous position.
    
    Uses PLL for each candidate, then normalizes via softmax to get
    probability-like scores that sum to 1 within the candidate set.
    
    Args:
        sentence_words: List of words (with None at ambiguous positions)
        position: Index of the current ambiguous position
        candidates: List of candidate words
        
    Returns:
        Dict mapping each candidate to its normalized MLM score [0, 1]
    """
    pll_scores = {}
    for candidate in candidates:
        pll_scores[candidate] = self._get_candidate_pll(
            sentence_words, position, candidate
        )
    
    # Normalize via softmax (converts log-likelihoods to probabilities)
    max_pll = max(pll_scores.values())
    exp_scores = {
        c: math.exp(s - max_pll) 
        for c, s in pll_scores.items()
    }
    total = sum(exp_scores.values())
    
    if total > 0:
        return {c: exp_scores[c] / total for c in candidates}
    else:
        # Fallback: equal scores
        return {c: 1.0 / len(candidates) for c in candidates}
```

The softmax normalization is given by:

$$\text{MLM score} = \frac{\exp(\text{PLL}_i)}{\sum_j \exp(\text{PLL}_j)}$$

The PLL approach provides a substantially stronger semantic signal than cosine similarity of word embeddings, particularly for rare words or morphological variants, because it directly interrogates the model's knowledge of contextual appropriateness rather than relying on similarity in a fixed embedding space.

## 3.3.2 Frequency-Based Scoring for No-Context Scenarios

When disambiguation must occur in contexts where insufficient surrounding words are available (i.e., when all other words in the input are also ambiguous), or as a fallback for unambiguous words, a frequency-based scoring mechanism is used. This approach relies on corpus statistics computed from the training corpora.

### Frequency Score Implementation

The frequency score is computed from corpus word counts as follows:

```python
def get_frequency_score(self, word: str) -> float:
    """
    Get normalized frequency score (0-1).
    Higher = more frequent word.
    """
    count = self.word_freq.get(word.lower(), 0)
    
    if count == 0:
        return 0.1  # Small non-zero for unknown words
    
    # Log-normalized by max frequency
    max_freq = self.word_freq.most_common(1)[0][1]
    return min(1.0, np.log(count + 1) / np.log(max_freq + 1))
```

The frequency score for a given word is computed as:

$$\text{Frequency Score}(w) = \frac{\log(\text{count}(w) + 1)}{\log(\text{max\_frequency} + 1)}$$

where $\text{count}(w)$ is the number of occurrences of word $w$ in the corpus and $\text{max\_frequency}$ is the count of the most frequent word. The logarithmic normalization prevents high-frequency words from entirely dominating the scores, and maps scores to the range [0, 1]. Words that do not appear in the corpus receive a small non-zero score (0.1) rather than zero, providing graceful degradation for out-of-vocabulary words.

### Corpus Loading

The corpus statistics are computed during initialization:

```python
def _load_corpora(self, text_files: List[str]):
    """Load word frequencies from text corpora."""
    all_words = []
    
    for filepath in text_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process sentence by sentence to enable exclusion
            sentences = re.split(r'[.!?]+\s+', text)
            file_words = 0
            
            for sentence in sentences:
                normalized = self._normalize_sentence(sentence)
                
                # Skip test sentences
                if normalized in self.excluded:
                    self.excluded_count += 1
                    continue
                
                # Extract words
                words = re.findall(r"[a-z\-']+", normalized)
                words = [w.strip("-'") for w in words if len(w) > 1]
                all_words.extend(words)
                file_words += len(words)
            
            print(f"  [OK] {Path(filepath).name}: {file_words:,} words")
            
        except FileNotFoundError:
            print(f"  [!] File not found: {filepath}")
    
    # Compute statistics
    self.word_freq = Counter(all_words)
    self.total_words = len(all_words)
    
    # Bigram frequencies
    for i in range(len(all_words) - 1):
        bigram = (all_words[i], all_words[i+1])
        self.bigram_freq[bigram] += 1
    self.total_bigrams = len(all_words) - 1
    
    print(f"  [OK] Total: {self.total_words:,} words, {len(self.word_freq):,} unique")
    if self.excluded_count > 0:
        print(f"  [OK] Excluded {self.excluded_count} test sentences")
```

The frequency score is computed once during corpus initialization and cached, making it computationally efficient during inference.

## 3.3.3 Disambiguation Strategy: MLM for Context, Frequency for Fallback

The implementation uses a two-tier strategy for selecting among candidate words:

1. **Ambiguous words (context available)**: Use MLM-PLL scoring exclusively to select the candidate that best fits the sentence context.
2. **No context (all words ambiguous)**: Fall back to frequency-based scoring using word prevalence in the corpus.

This is implemented in the main `disambiguate` method:

```python
def disambiguate(
    self,
    ocr_candidates: List[Union[str, List[str]]],
    ground_truth: str = None,
    use_mlm: bool = True
) -> Tuple[List[str], Dict]:
    """
    Disambiguate a sentence given OCR candidates.
    
    Strategy:
    - Ambiguous words: Use MLM-PLL scoring only (context-aware disambiguation)
    - Unambiguous words: Use frequency scoring only (fallback for single words)
    
    Args:
        ocr_candidates: List where each element is either:
            - str: unambiguous word
            - List[str]: ambiguous candidates to choose from
        ground_truth: Optional ground truth for context (used in evaluation)
        use_mlm: If True, use MLM PLL scoring for semantic feature.
                 If False, use cosine similarity of mean-pooled embeddings.
        
    Returns:
        Tuple of (disambiguated_words, debug_info)
    """
    # Build context from ground truth or UNAMBIGUOUS words only
    # (don't include ambiguous candidates to avoid bias toward first candidate)
    if ground_truth:
        context = ground_truth
    else:
        # Use only unambiguous words for context (skip ambiguous positions)
        context_words = [
            c if isinstance(c, str) else None
            for c in ocr_candidates
        ]
        # Filter out None and build context
        context = ' '.join(w for w in context_words if w is not None)
        # If all words are ambiguous, fall back to using first candidates
        if not context.strip():
            context = ' '.join(
                c[0] if isinstance(c, list) else c 
                for c in ocr_candidates
            )
    
    context_embedding = self.get_embedding(context)
    
    # ... [processing for each position] ...
    
    for pos, item in enumerate(ocr_candidates):
        if isinstance(item, list):
            # Ambiguous word: use MLM-PLL scoring
            if use_mlm:
                # Get MLM scores for all candidates (single set of forward passes)
                mlm_scores = self._get_mlm_scores(mlm_words, pos, item)
                
                # Score all candidates with MLM-enhanced semantic feature
                scores = {
                    c: self.score_candidate(
                        c, context_embedding, prev_word, next_word,
                        mlm_score=mlm_scores[c]
                    )
                    for c in item
                }
            
            # Select best candidate
            best = max(scores.keys(), key=lambda c: scores[c]['combined'])
            
            result.append(best)
            resolved[pos] = best
```

For the evaluation in this thesis, the feature weights are configured as:

```python
mlm_only_weights = {
    'semantic': 1.0,      # MLM-PLL scoring (context-aware)
    'frequency': 0.0,     # Disabled
    'cooccurrence': 0.0,  # Disabled
    'morphology': 0.0     # Disabled
}
```

With these settings, **only the MLM-PLL score is used** for ambiguous words; no weighted combination occurs. This prioritizes contextual appropriateness over other signals.

## 3.3.4 Tools and Technologies

The system was implemented in Python using the following key libraries:

- **PyTorch**: For efficient tensor operations and GPU acceleration during PLL computation
- **Hugging Face Transformers**: For loading pre-trained RoBERTa and accessing the MLM head
- **NumPy**: For numerical operations and score normalization

The RoBERTa model (`jcblaise/roberta-tagalog-base`) was used as the pre-trained language model, chosen because it is specifically trained on Tagalog text. This ensures the MLM head has strong knowledge of Filipino language context. All tensor operations and PLL computations are accelerated using CUDA when GPU hardware is available, with automatic fallback to CPU processing.

## 3.3.5 Baybayin Ambiguous Pairs

The ambiguous pairs were identified through a systematic analysis of the Filipino word corpus. The process involved analyzing 74,419+ words to find all ambiguous pairs that share identical Baybayin representations. Key functions from the ambiguous pair detection algorithm are shown below:

### Core Algorithm: Baybayin Conversion and Ambiguity Detection

The first step converts Latin script Filipino words to Baybayin representation to identify potential ambiguities:

```python
def latin_to_baybayin(text):
    """
    Converts Latin script Filipino text to Baybayin script.
    Identifies phonetic ambiguities by detecting where E/I and O/U distinctions collapse.
    """
    baybayin_chars = {
        # Independent vowels - note E and I both map to ᜁ, O and U both map to ᜂ
        'a': 'ᜀ', 'e': 'ᜁ', 'i': 'ᜁ', 'o': 'ᜂ', 'u': 'ᜂ',
        
        # Consonant + vowel combinations (simplified for core ambiguities)
        'ka': 'ᜃ', 'ki': 'ᜃᜒ', 'ke': 'ᜃᜒ', 'ko': 'ᜃᜓ', 'ku': 'ᜃᜓ',
        'ta': 'ᜆ', 'ti': 'ᜆᜒ', 'te': 'ᜆᜒ', 'to': 'ᜆᜓ', 'tu': 'ᜆᜓ',
        'da': 'ᜇ', 'di': 'ᜇᜒ', 'de': 'ᜇᜒ', 'do': 'ᜇᜓ', 'du': 'ᜇᜓ',
        'ra': 'ᜇ', 'ri': 'ᜇᜒ', 're': 'ᜇᜒ', 'ro': 'ᜇᜓ', 'ru': 'ᜇᜓ',
        # ... (additional consonants)
    }
    
    # Greedy left-to-right matching: prioritize longer matches (3-char, then 2-char, then 1-char)
    text = text.lower()
    result = []
    i = 0
    
    while i < len(text):
        matched = False
        
        # Try 3-character match first (e.g., 'nga', 'ngi')
        if i + 2 < len(text):
            three_char = text[i:i+3]
            if three_char in baybayin_chars:
                result.append(baybayin_chars[three_char])
                i += 3
                matched = True
                continue
        
        # Try 2-character match
        if i + 1 < len(text):
            two_char = text[i:i+2]
            if two_char in baybayin_chars:
                result.append(baybayin_chars[two_char])
                i += 2
                matched = True
                continue
        
        # Try 1-character match
        if text[i] in baybayin_chars:
            result.append(baybayin_chars[text[i]])
            i += 1
        else:
            result.append(text[i])  # Keep numbers and punctuation
            i += 1
    
    return ''.join(result)
```

### Classifying Ambiguity Types

Once words are mapped to Baybayin, their differences are analyzed to determine the type of ambiguity:

```python
def classify_ambiguity_type(words):
    """
    Determines what type of ambiguity causes these words to map to same Baybayin.
    Returns: 'E/I', 'O/U', 'D/R', 'COMBINED', or 'UNKNOWN'
    """
    has_e_i = False
    has_o_u = False
    has_d_r = False
    
    # Compare all word pairs to find character differences
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word1 = words[i].lower()
            word2 = words[j].lower()
            
            if len(word1) != len(word2):
                continue
            
            for c1, c2 in zip(word1, word2):
                if c1 != c2:
                    if (c1 in 'ei' and c2 in 'ei'):
                        has_e_i = True
                    elif (c1 in 'ou' and c2 in 'ou'):
                        has_o_u = True
                    elif (c1 in 'dr' and c2 in 'dr'):
                        has_d_r = True
    
    # Determine primary classification
    types = []
    if has_e_i:
        types.append('E/I')
    if has_o_u:
        types.append('O/U')
    if has_d_r:
        types.append('D/R')
    
    if len(types) == 0:
        return 'UNKNOWN'
    elif len(types) == 1:
        return types[0]
    else:
        return 'COMBINED'  # Multiple types present
```

### Finding and Filtering Ambiguous Groups

The main algorithm groups words by Baybayin representation and applies filters to remove false positives:

```python
def find_ambiguous_groups(words, proper_nouns, dictionary=None):
    """
    Groups words by Baybayin representation and filters for genuine ambiguities.
    Removes: (1) duplicate words, (2) proper-noun-only words, (3) non-dictionary words
    """
    from collections import defaultdict
    
    # Map each word to its Baybayin representation
    baybayin_map = defaultdict(list)
    for word in words:
        baybayin = latin_to_baybayin(word)
        baybayin_map[baybayin].append(word)
    
    ambiguous_groups = {}
    
    for baybayin, word_list in baybayin_map.items():
        total_occurrences = len(word_list)
        
        # Get unique words (case-insensitive), filter out proper-noun-only words
        seen_lower = {}
        for word in word_list:
            lower = word.lower()
            # Skip words that ONLY appear capitalized (proper nouns)
            if lower in proper_nouns:
                continue
            if lower not in seen_lower:
                seen_lower[lower] = word.lower()
        
        unique_words = list(seen_lower.values())
        
        # Dictionary filtering: keep group only if at least one word is valid
        valid_words = []
        if dictionary is not None:
            for word in unique_words:
                if word.lower() in dictionary:
                    valid_words.append(word)
            
            if len(valid_words) == 0 and len(unique_words) > 0:
                continue  # Skip if no valid dictionary words
        else:
            valid_words = unique_words
        
        # Keep if: (1) multiple valid words, OR (2) at least 1 valid + other candidates
        if len(valid_words) > 1 or (len(valid_words) >= 1 and len(unique_words) > 1):
            ambiguous_groups[baybayin] = {
                'unique_words': unique_words,
                'total_occurrences': total_occurrences,
                'valid_words': valid_words
            }
    
    return ambiguous_groups
```

### Results

This algorithm analyzed the 74,419+ word corpus and generated a comprehensive CSV file (`ambiguous_pairs_complete.csv`) containing all discovered ambiguous pairs, their Baybayin representations, ambiguity types (E/I, O/U, D/R), and frequency statistics.

From this complete candidate list, we manually selected the **15 most common pairs** based on frequency and relevance to the Baybayin OCR task:

- asero/asido (E/I)
- bote/buti (O/U)  
- boto/buto (O/U)
- higante/higanti (E/I)
- hito/heto (E/I)
- itodo/ituro (O/U + D/R)
- kompas/kumpas (O/U)
- kumita/kometa (O/U + E/I)
- mesa/misa (E/I)
- polo/pulo (O/U)
- poso/puso (O/U)
- tela/tila (E/I)
- todo/toro/turo (O/U + D/R)
- toyo/tuyo (O/U)
- kamada/kamara (D/R)

These 15 pairs became the focus of the evaluation and model development efforts.

To facilitate efficient data collection and tracking, a website was created to manage the sentence creation workflow for each ambiguous pair. This tool provided a centralized interface for: (1) viewing the selected pairs, (2) creating and validating tagged sentences for each pair, and (3) tracking progress across the dataset creation process.

## 3.3.6 Standardized Evaluation Dataset

The evaluation dataset is housed in the `gold_standard_dataset` directory, which contains the standardized evaluation data used to assess the disambiguation system. The dataset includes:

- **sentences/**: Ground truth sentences categorized by ambiguous pair, with manual annotations for each word's correct category
- **images/**: Baybayin images organized by pair (e.g., `01_asido_asero/`), containing the actual handwritten/stylized text that serves as OCR input
- **results/**: Evaluation results and performance metrics from running the disambiguation system on the gold standard data
- **tracking.md**: Documentation of the dataset creation process and versioning history

The gold_standard_dataset provides a rigorous benchmark for evaluating the MLM-PLL disambiguation approach against manually validated ground truth, ensuring reproducible and comparable results across different model configurations and feature combinations.

---

# Chapter 3.4: Testing and Evaluation

The implemented disambiguation system was rigorously tested through a multi-level evaluation strategy designed to assess both per-pair performance and system-wide effectiveness. Testing was conducted on the gold standard dataset using the 15 manually curated ambiguous pairs and their associated sentence sets.

## 3.4.1 Testing Strategy and Evaluation Framework

A comprehensive evaluation framework was developed to assess the disambiguation system across three dimensions: (1) **unit testing** of individual ambiguous pairs, (2) **comprehensive system evaluation** comparing different feature configurations, and (3) **baseline comparisons** against naive and embedding-based approaches.

### Test Data Structure

Test data for each ambiguous pair was structured to enable systematic performance measurement. Each test case consists of:

```python
{
    'sentence': str,              # Ground truth sentence with target word
    'ground_truth': str,          # Original sentence (for evaluation reference)
    'ocr_candidates': [           # OCR candidate words (one per position)
        'word1',                  # Unambiguous word (single candidate)
        ['word_a', 'word_b'],     # Ambiguous word (multiple candidates)
        'word3',                  # ...
    ]
}
```

For each ambiguous pair, 100 test sentences were created (50 containing each variant word), providing balanced evaluation. The test dataset was completely isolated from the corpus used for training the language model and computing frequency statistics, ensuring an unbiased evaluation.

## 3.4.2 Unit Testing: Individual Ambiguous Pair Evaluation

Each of the 15 ambiguous pairs was evaluated independently using a dedicated test script. These scripts follow a consistent pattern: loading ground truth sentences, creating synthetic OCR candidate lists, and measuring disambiguation accuracy.

### Test Execution Example: ASERO/ASIDO

The following code illustrates the unit test structure for the asero/asido pair:

```python
def parse_sentence_file(filepath):
    """Parse sentence file and separate by target words (case-insensitive)."""
    asero_sentences = []
    asido_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Separate based on exact word match
    for line in lines:
        words = re.findall(r'\b\w+\b', line.lower())
        if "asero" in words:
            asero_sentences.append(line)
        elif "asido" in words:
            asido_sentences.append(line)
    
    return asero_sentences, asido_sentences

# Load test data
asero_sentences, asido_sentences = parse_sentence_file("gold_standard_dataset/sentences/01_asido_asero.txt")

# Create synthetic OCR candidates where the target word is ambiguous
test_data = []
for sent in asero_sentences + asido_sentences:
    words = sent.lower().split()
    ocr_candidates = []
    for word in words:
        if word in ["asero", "asido"]:
            # Mark as ambiguous with both candidates
            ocr_candidates.append(["asero", "asido"])
        else:
            # Keep unambiguous words as single candidates
            ocr_candidates.append(word)
    
    test_data.append({
        'sentence': sent,
        'ground_truth': sent,
        'ocr_candidates': ocr_candidates
    })

# Evaluate
disambiguator = BaybayinDisambiguator()
metrics, details = disambiguator.evaluate(test_data, use_mlm=True)
```

This unit test structure was replicated for all 15 pairs, enabling independent assessment of disambiguation performance per pair and across ambiguity types (E/I, O/U, D/R).

## 3.4.3 Per-Pair Test Execution and Result Aggregation

Individual test scripts were created for each of the 15 ambiguous pairs, located in the `tests/` directory (e.g., `test_asero_asido.py`, `test_bote_buti.py`, etc.). Each script follows the pattern shown in section 3.4.2, executes the disambiguation evaluation, and saves detailed metrics to a JSON file in `gold_standard_dataset/results/`.

### Running Tests and Collecting Results

Each test script independently:
1. Loads ground truth sentences for the pair from `gold_standard_dataset/sentences/`
2. Creates synthetic OCR candidate lists marking the target word as ambiguous
3. Runs the BaybayinDisambiguator on the test data
4. Computes accuracy metrics
5. Saves results to `gold_standard_dataset/results/results_{pair}.json`

The JSON results file for each pair captures:

```json
{
  "ambiguous_pair": "word_a, word_b",
  "baybayin": "ᜀᜐᜒᜇᜓ",
  "type": "E/I + O/U",
  "test_sentences": 100,
  "comparison": {
    "baseline": {
      "name": "MaBaybay Default (First Candidate)",
      "accuracy": 50.0,
      "correct": 50,
      "word_a_accuracy": "50/50",
      "word_b_accuracy": "0/50"
    },
    "mlm_pll": {
      "name": "Pure MLM-PLL (MLM Scoring Only)",
      "accuracy": 59.0,
      "correct": 59,
      "word_a_accuracy": "9/50",
      "word_b_accuracy": "50/50"
    },
    "improvement_over_baseline": 9.0
  },
  "metrics": {
    "total_words": 962,
    "correct_words": 921,
    "total_accuracy": 0.9574,
    "total_ambiguous": 100,
    "correct_ambiguous": 59,
    "ambiguous_accuracy": 0.59,
    "timing_per_sample_ms": 248.1
  }
}
```

### Key Metrics in Results

- **Ambiguous Accuracy**: Accuracy only on the target ambiguous word position (the metric of interest for disambiguation effectiveness)
- **Total Accuracy**: Overall word accuracy including all unambiguous words (~96% as baseline)
- **Improvement Over Baseline**: Percentage point improvement of MLM-PLL over MaBaybay Default
- **Per-Variant Accuracy**: Separate accuracy for each word variant (e.g., asero vs asido)

## 3.4.4 Baseline: MaBaybay Default (First Candidate Selection)

The baseline approach represents the original MaBaybay-OCR behavior: always selecting the first candidate from the transliteration output, regardless of context. For any ambiguous pair, MaBaybay's transliteration order is fixed (e.g., `["asero", "asido"]` with "asero" first), making the baseline accuracy 50% by definition.

The baseline computation is straightforward:

```python
# BASELINE: MaBaybay Default (First Candidate Selection)
baseline_correct_total = 0
baseline_correct_asero = 0
baseline_correct_asido = 0

for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    # Baseline always picks first candidate (e.g., "asero")
    if "asero" in gt_words:
        baseline_correct_total += 1
        baseline_correct_asero += 1
    # If ground truth is "asido", baseline gets it wrong (picks "asero")

baseline_accuracy = baseline_correct_total / 100 * 100

print(f"Baseline Strategy: Always select '{first_candidate}'")
print(f"Overall baseline accuracy: {baseline_correct_total}/100 = {baseline_accuracy:.2f}%")
```

## 3.4.5 Main Evaluation: Pure MLM-PLL Disambiguation

Once baselines are established, the BaybayinDisambiguator is initialized and evaluated using pure MLM-PLL scoring (semantic context only). This represents the full implementation with no feature simplification:

```python
# INITIALIZE MODEL
model = BaybayinDisambiguator(
    corpus_files=[
        "corpus/Tagalog_Literary_Text.txt",
        "corpus/Tagalog_Religious_Text.txt",
        "corpus/Tagalog_Balita_Texts_Balanced.txt"
    ],
    exclude_sentences=all_test_sentences  # Clean evaluation - no data leakage
)

# EVALUATE: Pure MLM-PLL
mlm_only_weights = {
    'semantic': 1.0,      # MLM-PLL scoring enabled
    'frequency': 0.0,     # Disabled
    'cooccurrence': 0.0,  # Disabled
    'morphology': 0.0     # Disabled
}

mlm_only_metrics, mlm_only_results = model.evaluate(
    test_data, 
    show_progress=True, 
    use_mlm=True, 
    weights_override=mlm_only_weights
)

mlm_only_accuracy = mlm_only_metrics['ambiguous_accuracy'] * 100
improvement = mlm_only_accuracy - baseline_accuracy

print(f"MLM-PLL accuracy: {mlm_only_accuracy:.2f}%")
print(f"Improvement over baseline: {improvement:+.2f} percentage points")
```

Per-word accuracy is then computed to show performance on each variant:

```python
def count_word_accuracy(result_list, word1, word2):
    """Count correct predictions for each word variant."""
    w1_correct = 0
    w2_correct = 0
    for test_item, result_item in zip(test_data, result_list):
        gt_words = get_clean_words(test_item['ground_truth'])
        pred_words = get_clean_words(result_item['predicted'])
        if word1 in gt_words:
            if word1 in pred_words:
                w1_correct += 1
        elif word2 in gt_words:
            if word2 in pred_words:
                w2_correct += 1
    return w1_correct, w2_correct

mlm_asero_correct, mlm_asido_correct = count_word_accuracy(mlm_only_results, "asero", "asido")
```

## 3.4.6 Results Summary and Comparison

Results are compared in a summary table showing baseline vs MLM-PLL performance:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       DISAMBIGUATION RESULTS                             │
├──────────────────────────────┬──────────────┬────────────┬───────────────┤
│        Method                │   Accuracy   │ Word_A     │  Word_B       │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ MaBaybay Default             │    50.00%    │   50/50    │    0/50       │
│ (First Candidate)            │              │            │               │
├──────────────────────────────┼──────────────┼────────────┼───────────────┤
│ Pure MLM-PLL                 │    59.00%    │    9/50    │   50/50       │
│ (MLM Scoring Only)           │  (+9.00%)    │            │               │
└──────────────────────────────┴──────────────┴────────────┴───────────────┘
```

Each test script saves detailed results to JSON format for aggregation across all 15 pairs:

```python
output = {
    'ambiguous_pair': 'asero, asido',
    'baybayin': 'ᜀᜐᜒᜇᜓ',
    'type': 'E/I + O/U',
    'test_sentences': 100,
    'comparison': {
        'baseline': {
            'name': 'MaBaybay Default (First Candidate)',
            'accuracy': baseline_accuracy,
            'asero_accuracy': f"{baseline_correct_asero}/50",
            'asido_accuracy': f"{baseline_correct_asido}/50"
        },
        'mlm_pll': {
            'name': 'Pure MLM-PLL (MLM Scoring Only)',
            'accuracy': mlm_only_accuracy,
            'asero_accuracy': f"{mlm_asero_correct}/50",
            'asido_accuracy': f"{mlm_asido_correct}/50"
        },
        'improvement_over_baseline': improvement
    },
    'metrics': mlm_only_metrics
}

# Save to gold_standard_dataset/results/results_{pair}.json
with open(f"gold_standard_dataset/results/results_{pair_name}.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
```

Results from all 15 pairs are automatically saved to the `/results/` directory and can be reviewed for per-pair performance analysis. The JSON structure enables easy aggregation and visualization of results across all ambiguity types.

