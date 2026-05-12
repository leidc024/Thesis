# Appendix: Source Code

## A. Core Disambiguation Engine

```python
class BaybayinDisambiguator:
    def __init__(
        self,
        model_name: str = "jcblaise/roberta-tagalog-base",
        corpus_files: List[str] = None,
        exclude_sentences: List[str] = None,
        weights: Dict[str, float] = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights or DEFAULT_WEIGHTS
        corpus_files = corpus_files or DEFAULT_CORPORA
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.corpus = CorpusStatistics(
            text_files=corpus_files,
            exclude_sentences=exclude_sentences
        )
        
        self.morphology = MorphologicalAnalyzer()
```

**Configuration:**

```python
DEFAULT_MODEL = "jcblaise/roberta-tagalog-base"

DEFAULT_WEIGHTS = {
    'semantic': 1.0,
    'frequency': 0.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}

## B. Pseudo-Log-Likelihood Scoring

```python
def _get_candidate_pll(
    self,
    sentence_words: List[str],
    position: int,
    candidate: str
) -> float:
    words = list(sentence_words)
    words[position] = candidate
    full_sentence = ' '.join(words)
    
    encoding = self.tokenizer(
        full_sentence,
        return_tensors='pt',
        truncation=True,
        max_length=128
    )
    token_ids = encoding['input_ids'][0].clone().to(self.device)
    attention_mask = encoding['attention_mask'][0].to(self.device)
    
    cand_tokens = self.tokenizer.encode(' ' + candidate, add_special_tokens=False)
    cand_positions = self._find_subtoken_positions(token_ids.cpu(), cand_tokens)
    
    if not cand_positions:
        cand_tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
        cand_positions = self._find_subtoken_positions(token_ids.cpu(), cand_tokens)
    
    if not cand_positions:
        return -100.0
    
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

```python
def _get_mlm_scores(
    self,
    sentence_words: List[str],
    position: int,
    candidates: List[str]
) -> Dict[str, float]:
    pll_scores = {}
    for candidate in candidates:
        pll_scores[candidate] = self._get_candidate_pll(
            sentence_words, position, candidate
        )
    
    max_pll = max(pll_scores.values())
    exp_scores = {
        c: math.exp(s - max_pll) 
        for c, s in pll_scores.items()
    }
    total = sum(exp_scores.values())
    
    if total > 0:
        return {c: exp_scores[c] / total for c in candidates}
    else:
        return {c: 1.0 / len(candidates) for c in candidates}
```

```python
def _find_subtoken_positions(
    self,
    token_ids: torch.Tensor,
    subtoken_ids: List[int]
) -> List[int]:
    seq = token_ids.tolist()
    sub = subtoken_ids
    n = len(sub)
    
    for i in range(len(seq) - n + 1):
        if seq[i:i+n] == sub:
            return list(range(i, i + n))
    
    return []
```

## C. Corpus Statistics Module

```python
class CorpusStatistics:
    def __init__(
        self, 
        text_files: List[str],
        exclude_sentences: List[str] = None,
        vocab_file: str = None
    ):
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.total_words = 0
        self.total_bigrams = 0
        self.vocab: Set[str] = set()
        self.excluded_count = 0
        
        if exclude_sentences:
            self.excluded = set(
                self._normalize_sentence(s) for s in exclude_sentences
            )
        else:
            self.excluded = set()
        
        if vocab_file:
            self._load_vocab(vocab_file)
        
        self._load_corpora(text_files)
```

## D. Frequency-Based Scoring

```python
def get_frequency_score(self, word: str) -> float:
    count = self.word_freq.get(word.lower(), 0)
    
    if count == 0:
        return 0.1
    
    max_freq = self.word_freq.most_common(1)[0][1]
    return min(1.0, np.log(count + 1) / np.log(max_freq + 1))
```

## E. Bigram Probability

```python
def get_bigram_probability(self, word1: str, word2: str) -> float:
    bigram = (word1.lower(), word2.lower())
    bigram_count = self.bigram_freq.get(bigram, 0)
    word1_count = self.word_freq.get(word1.lower(), 0)
    
    if word1_count == 0:
        return 0.0
    
    vocab_size = len(self.word_freq)
    return (bigram_count + 1) / (word1_count + vocab_size)
```

## F. Corpus Loading

```python
def _load_corpora(self, text_files: List[str]):
    all_words = []
    
    for filepath in text_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            sentences = re.split(r'[.!?]+\s+', text)
            file_words = 0
            
            for sentence in sentences:
                normalized = self._normalize_sentence(sentence)
                
                if normalized in self.excluded:
                    self.excluded_count += 1
                    continue
                
                words = re.findall(r"[a-z\-']+", normalized)
                words = [w.strip("-'") for w in words if len(w) > 1]
                all_words.extend(words)
                file_words += len(words)
            
            print(f"  [OK] {Path(filepath).name}: {file_words:,} words")
            
        except FileNotFoundError:
            print(f"  [!] File not found: {filepath}")
    
    self.word_freq = Counter(all_words)
    self.total_words = len(all_words)
    
    for i in range(len(all_words) - 1):
        bigram = (all_words[i], all_words[i+1])
        self.bigram_freq[bigram] += 1
    self.total_bigrams = len(all_words) - 1
    
    print(f"  [OK] Total: {self.total_words:,} words, {len(self.word_freq):,} unique")
    if self.excluded_count > 0:
        print(f"  [OK] Excluded {self.excluded_count} test sentences")
```

## G. Morphological Analysis

```python
class MorphologicalAnalyzer:
    PREFIXES = [
        'nakapag', 'makapag', 'nakaka', 'mapag', 'ipang', 'ipag', 'maki',
        'maka', 'taga', 'sang', 'tag',
        'mag', 'nag', 'pag', 'ika', 'ipa',
        'um', 'in', 'ka', 'ma', 'na', 'pa', 'i'
    ]
    
    INFIXES = ['um', 'in']
    
    SUFFIXES = [
        'han', 'hin', 'nan', 'ang', 'ing',
        'an', 'in', 'ng'
    ]
    
    VALID_ENDINGS = ('a', 'e', 'i', 'o', 'u', 'ng', 'n', 'g', 'y', 'w')
```

## H. Morphological Scoring

```python
def get_morphological_score(self, word: str) -> float:
    word = word.lower()
    score = 0.5
    
    for prefix in self.PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            score += 0.1
            break
    
    for suffix in self.SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            score += 0.1
            break
    
    if self.REDUPLICATION_PATTERN.match(word):
        score += 0.1
    
    if len(word) <= 2:
        score -= 0.1
    
    if word.endswith(self.VALID_ENDINGS):
        score += 0.05
    
    if self._has_common_pattern(word):
        score += 0.05
    
    return min(1.0, max(0.0, score))
```

## I. Root Word Extraction

```python
def get_root_word(self, word: str) -> str:
    word = word.lower()
    
    for prefix in self.PREFIXES:
        if word.startswith(prefix):
            word = word[len(prefix):]
            break
    
    for suffix in self.SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[:-len(suffix)]
            break
    
    return word
```

## J. Multi-Feature Scoring

```python
def score_candidate(
    self,
    candidate: str,
    context_embedding: np.ndarray,
    prev_word: Optional[str] = None,
    next_word: Optional[str] = None,
    mlm_score: Optional[float] = None
) -> Dict[str, float]:
    scores = {}
    
    if mlm_score is not None:
        scores['semantic'] = mlm_score
    else:
        cand_emb = self.get_embedding(candidate)
        semantic_sim = cosine_similarity(
            cand_emb.reshape(1, -1),
            context_embedding.reshape(1, -1)
        )[0, 0]
        scores['semantic'] = max(0.0, float(semantic_sim))
    
    scores['frequency'] = self.corpus.get_frequency_score(candidate)
    
    cooc = 0.0
    if prev_word:
        cooc += self.corpus.get_bigram_probability(prev_word, candidate)
    if next_word:
        cooc += self.corpus.get_bigram_probability(candidate, next_word)
    scores['cooccurrence'] = min(1.0, cooc * 10)
    
    scores['morphology'] = self.morphology.get_morphological_score(candidate)
    
    scores['combined'] = (
        self.weights['semantic'] * scores['semantic'] +
        self.weights['frequency'] * scores['frequency'] +
        self.weights['cooccurrence'] * scores['cooccurrence'] +
        self.weights['morphology'] * scores['morphology']
    )
    
    return scores
```

## K. Main Disambiguation Method

```python
def disambiguate(
    self,
    ocr_candidates: List[Union[str, List[str]]],
    ground_truth: str = None,
    use_mlm: bool = True
) -> Tuple[List[str], Dict]:
    if ground_truth:
        context = ground_truth
    else:
        context_words = [
            c if isinstance(c, str) else None
            for c in ocr_candidates
        ]
        context = ' '.join(w for w in context_words if w is not None)
        if not context.strip():
            context = ' '.join(
                c[0] if isinstance(c, list) else c 
                for c in ocr_candidates
            )
    
    context_embedding = self.get_embedding(context)
    
    sentence_words = []
    for c in ocr_candidates:
        if isinstance(c, str):
            sentence_words.append(c)
        else:
            sentence_words.append(None)
    
    resolved = [
        None if isinstance(c, list) else c 
        for c in ocr_candidates
    ]
    
    result = []
    debug = {'selected': {}, 'scores': {}}
    
    for pos, item in enumerate(ocr_candidates):
        if isinstance(item, list):
            prev_word = resolved[pos - 1] if pos > 0 else None
            next_word = None
            for j in range(pos + 1, len(ocr_candidates)):
                if not isinstance(ocr_candidates[j], list):
                    next_word = ocr_candidates[j]
                    break
            
            if use_mlm:
                mlm_words = list(sentence_words)
                for i in range(len(mlm_words)):
                    if mlm_words[i] is None and i != pos:
                        if resolved[i]:
                            mlm_words[i] = resolved[i]
                        elif isinstance(ocr_candidates[i], list):
                            mlm_words[i] = ocr_candidates[i][0]
                
                mlm_scores = self._get_mlm_scores(mlm_words, pos, item)
                
                scores = {
                    c: self.score_candidate(
                        c, context_embedding, prev_word, next_word,
                        mlm_score=mlm_scores[c]
                    )
                    for c in item
                }
            else:
                scores = {
                    c: self.score_candidate(
                        c, context_embedding, prev_word, next_word
                    )
                    for c in item
                }
            
            best = max(scores.keys(), key=lambda c: scores[c]['combined'])
            
            result.append(best)
            resolved[pos] = best
            debug['selected'][pos] = best
            debug['scores'][pos] = {c: s['combined'] for c, s in scores.items()}
        else:
            freq_score = self.corpus.get_frequency_score(item)
            result.append(item)
    
    return result, debug
```

## L. MATLAB Integration Wrapper

```python
"""
Disambiguate MaBaybay OCR candidates.
Called from MATLAB: system('python disambiguate.py input.json')
"""

import sys
import json
import os
from pathlib import Path

def disambiguate(candidates_json: str) -> str:
    import io
    import warnings
    
    warnings.filterwarnings('ignore')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TQDM_DISABLE'] = '1'
    
    script_dir = Path(__file__).parent.resolve()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        from src.disambiguator import BaybayinDisambiguator
        
        with open(candidates_json, 'r', encoding='utf-8') as f:
            raw_candidates = json.load(f)
        
        candidates = []
        for item in raw_candidates:
            if isinstance(item, list) and len(item) == 1:
                candidates.append(item[0])
            else:
                candidates.append(item)
        
        num_ambiguous = sum(1 for c in candidates if isinstance(c, list))
        num_words = len(candidates)
        
        model = BaybayinDisambiguator(
            corpus_files=[
                str(script_dir / "corpus" / "Tagalog_Literary_Text.txt"),
                str(script_dir / "corpus" / "Tagalog_Religious_Text.txt"),
                str(script_dir / "corpus" / "Tagalog_Balita_Texts_Balanced.txt")
            ]
        )
        
        result, _ = model.disambiguate(candidates)
        
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return ' '.join(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python disambiguate.py candidates.json", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        result = disambiguate(input_file)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

## M. Baseline Models

```python
class MaBaybayDefault:
    def __init__(self):
        print("[MaBaybay Default] Initialized (always selects first candidate)")
    
    def disambiguate(
        self,
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        result = []
        for item in ocr_candidates:
            if isinstance(item, list):
                result.append(item[0])
            else:
                result.append(item)
        return result, {}
```

## N. Embedding-Only Baseline

```python
class EmbeddingOnly:
    def __init__(self, model_name: str = "jcblaise/roberta-tagalog-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedding-Only] Loading RoBERTa on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[Embedding-Only] Initialized")
    
    def get_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
            mean_emb = (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return mean_emb.cpu().numpy().flatten()
    
    def disambiguate(
        self,
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        context = ground_truth if ground_truth else ' '.join(
            c[0] if isinstance(c, list) else c for c in ocr_candidates
        )
        context_emb = self.get_embedding(context)
        
        result = []
        debug = {}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                scores = {}
                for cand in item:
                    cand_emb = self.get_embedding(cand)
                    sim = cosine_similarity(
                        cand_emb.reshape(1, -1),
                        context_emb.reshape(1, -1)
                    )[0, 0]
                    scores[cand] = float(sim)
                
                best = max(scores.keys(), key=lambda c: scores[c])
                result.append(best)
                debug[pos] = scores
            else:
                result.append(item)
        
        return result, debug
```

## O. LLM-Based Baseline

```python
class LLMBaseline:
    def __init__(self, provider: str = "gemini", model: str = None):
        self.provider = provider.lower()
        self.client = None
        
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-3.5-turbo"
            print(f"[LLM Baseline] Using OpenAI {self.model}")
                
        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.model = model or "gemini-2.0-flash"
            self.client = genai.GenerativeModel(self.model)
            print(f"[LLM Baseline] Using Google Gemini {self.model}")
                
        elif self.provider == "ollama":
            import ollama
            self.client = ollama
            self.model = model or "llama3.2"
            print(f"[LLM Baseline] Using Ollama {self.model}")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _query_llm(self, prompt: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)
            return response.text
        elif self.provider == "ollama":
            response = self.client.generate(model=self.model, prompt=prompt)
            return response['response']
    
    def disambiguate(
        self,
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        result = []
        debug = {}
        
        ambiguous_positions = []
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list) and len(item) > 1:
                ambiguous_positions.append((pos, item))
        
        if not ambiguous_positions:
            for item in ocr_candidates:
                result.append(item[0] if isinstance(item, list) else item)
            return result, debug
        
        context_words = []
        for i, item in enumerate(ocr_candidates):
            if isinstance(item, list) and len(item) > 1:
                amb_idx = next(j for j, (p, _) in enumerate(ambiguous_positions) if p == i)
                context_words.append(f"[{amb_idx + 1}]")
            else:
                context_words.append(item[0] if isinstance(item, list) else item)
        
        sentence_context = " ".join(context_words)
        
        candidates_list = []
        for idx, (pos, candidates) in enumerate(ambiguous_positions):
            cand_str = ", ".join(f'"{c}"' for c in candidates)
            candidates_list.append(f"  [{idx + 1}]: {cand_str}")
        
        candidates_prompt = "\n".join(candidates_list)
        
        prompt = f"""You are an expert linguist specializing in Filipino/Tagalog language.

Choose the CORRECT Filipino word for each position marked with brackets.

Sentence: "{sentence_context}"

Candidates for each position:
{candidates_prompt}

Reply with ONLY the words separated by commas, nothing else."""
        
        response = self._query_llm(prompt)
        debug['prompt'] = prompt
        debug['response'] = response
        
        response_words = [w.strip().strip('"\'.,') for w in response.split(',')]
        
        amb_idx = 0
        for i, item in enumerate(ocr_candidates):
            if isinstance(item, list) and len(item) > 1:
                if amb_idx < len(response_words):
                    llm_choice = response_words[amb_idx].lower()
                    selected = item[0]
                    for cand in item:
                        if cand.lower() == llm_choice:
                            selected = cand
                            break
                    result.append(selected)
                else:
                    result.append(item[0])
                amb_idx += 1
            else:
                result.append(item[0] if isinstance(item, list) else item)
        
        return result, debug
```

## P. Evaluation Framework

```python
def evaluate(
    self,
    test_data: List[Dict],
    show_progress: bool = True,
    use_ground_truth_context: bool = False,
    use_mlm: bool = True,
    weights_override: Dict[str, float] = None
) -> Tuple[Dict, List]:
    total_words = 0
    correct_words = 0
    total_ambiguous = 0
    correct_ambiguous = 0
    results = []
    
    original_weights = self.weights
    if weights_override:
        self.weights = weights_override
    
    try:
        iterator = tqdm(test_data, desc="Evaluating") if show_progress else test_data
        
        for entry in iterator:
            gt = entry['ground_truth']
            candidates = entry['ocr_candidates']
            gt_words = gt.lower().split()
            
            predicted, debug = self.disambiguate(
                candidates,
                ground_truth=gt if use_ground_truth_context else None,
                use_mlm=use_mlm
            )
            
            for i, (pred, gt_word) in enumerate(zip(predicted, gt_words)):
                if i >= len(candidates):
                    break
                
                is_ambiguous = isinstance(candidates[i], list)
                is_correct = pred.lower() == gt_word.lower()
                
                total_words += 1
                if is_correct:
                    correct_words += 1
                
                if is_ambiguous:
                    total_ambiguous += 1
                    if is_correct:
                        correct_ambiguous += 1
            
            results.append({
                'gt': gt,
                'pred': ' '.join(predicted),
                'debug': debug
            })
        
        metrics = {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous else 0
        }
        
    finally:
        self.weights = original_weights
    
    return metrics, results
```

## Q. Configuration Constants

```python
DEFAULT_MODEL = "jcblaise/roberta-tagalog-base"

DEFAULT_WEIGHTS = {
    'semantic': 1.0,
    'frequency': 0.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}

FREQUENCY_ONLY_WEIGHTS = {
    'semantic': 0.0,
    'frequency': 1.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}

DEFAULT_CORPORA = [
    "corpus/Tagalog_Literary_Text.txt",
    "corpus/Tagalog_Religious_Text.txt",
    "corpus/Tagalog_Balita_Texts_Balanced.txt"
]

MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 32

PREFIXES = [
    'nakapag', 'makapag', 'nakaka', 'mapag', 'ipang', 'ipag', 'maki',
    'maka', 'taga', 'sang', 'tag', 'mag', 'nag', 'pag', 'ika', 'ipa',
    'um', 'in', 'ka', 'ma', 'na', 'pa', 'i'
]

SUFFIXES = [
    'han', 'hin', 'nan', 'ang', 'ing', 'an', 'in', 'ng'
]

VALID_ENDINGS = ('a', 'e', 'i', 'o', 'u', 'ng', 'n', 'g', 'y', 'w')
```

## R. MATLAB Integration Layer

```matlab
% ============================================================
% disambiguate_candidates.m
% Use Python disambiguation model instead of defaulting to first candidate
% ============================================================

function result = disambiguate_candidates(transliterations)
    thesis_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..');
    python_script = fullfile(thesis_dir, 'disambiguate.py');
    python_exe = fullfile(thesis_dir, '.venv', 'Scripts', 'python.exe');
    temp_json = fullfile(tempdir, 'mabaybay_candidates.json');
    
    candidates = cell(1, length(transliterations));
    for i = 1:length(transliterations)
        word_cands = transliterations{i};
        if ischar(word_cands)
            candidates{i} = {word_cands};
        elseif iscell(word_cands)
            candidates{i} = word_cands(:)';
        else
            candidates{i} = {char(word_cands)};
        end
    end
    
    json_str = jsonencode(candidates);
    fid = fopen(temp_json, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    fprintf('Running context-aware disambiguation (may take 15-30 sec on first run)...\n');
    fprintf('JSON saved to: %s\n', temp_json);
    fprintf('Candidates JSON: %s\n', json_str);
    python_cmd = sprintf('"%s" "%s" "%s"', python_exe, python_script, temp_json);
    [status, output] = system(python_cmd);
    
    fprintf('Python status: %d\n', status);
    fprintf('Python output: [%s]\n', output);
    
    if status ~= 0
        warning('Disambiguation failed, using first candidates. Error: %s', output);
        result = cell(1, length(transliterations));
        for i = 1:length(transliterations)
            word_cands = transliterations{i};
            if iscell(word_cands)
                result{i} = word_cands{1};
            else
                result{i} = char(word_cands);
            end
        end
        return;
    end
    
    output = strtrim(output);
    result = strsplit(output, ' ');
    
    delete(temp_json);
end
```

## S. Testing Framework

```python
"""
Test disambiguator on word pairs
"""

import json
import re
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.disambiguator import BaybayinDisambiguator

def get_clean_words(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    return words

def parse_sentence_file(filepath):
    bote_sentences = []
    buti_sentences = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        words = get_clean_words(line)
        if "bote" in words:
            bote_sentences.append(line)
        elif "buti" in words:
            buti_sentences.append(line)
    
    return bote_sentences, buti_sentences

SENTENCE_FILE = "gold_standard_dataset/sentences/03_bote_buti.txt"
bote_sentences, buti_sentences = parse_sentence_file(SENTENCE_FILE)

test_data = []

for sent in bote_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word == "bote":
            candidates.append(["bote", "buti"])
        else:
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

for sent in buti_sentences:
    words = sent.split()
    candidates = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word == "buti":
            candidates.append(["bote", "buti"])
        else:
            candidates.append(clean_word)
    
    test_data.append({
        'ground_truth': sent,
        'ocr_candidates': candidates
    })

print(f"{'='*70}")
print("BASELINE: MaBaybay Default (First Candidate)")
print(f"{'='*70}")

baseline_correct_total = 0
baseline_correct_bote = 0
baseline_correct_buti = 0

baseline_start_time = time.time()
for test_item in test_data:
    gt = test_item['ground_truth']
    gt_words = get_clean_words(gt)
    
    if "bote" in gt_words:
        baseline_correct_total += 1
        baseline_correct_bote += 1

baseline_time = time.time() - baseline_start_time
baseline_accuracy = baseline_correct_total / 100 * 100

print(f"Bote accuracy: {baseline_correct_bote}/50 = {baseline_correct_bote/50:.2%}")
print(f"Buti accuracy: {baseline_correct_buti}/50 = {baseline_correct_buti/50:.2%}")
print(f"Overall accuracy: {baseline_correct_total}/100 = {baseline_accuracy:.2f}%")
print(f"Execution time: {baseline_time*1000:.2f}ms ({(baseline_time/100)*1000:.4f}ms per position)")

print(f"\n{'='*70}")
print("INITIALIZING MODEL")
print(f"{'='*70}")

all_test_sentences = [item['ground_truth'] for item in test_data]
init_start_time = time.time()
model = BaybayinDisambiguator(
    corpus_files=[
        "corpus/Tagalog_Literary_Text.txt",
        "corpus/Tagalog_Religious_Text.txt",
        "corpus/Tagalog_Balita_Texts_Balanced.txt"
    ],
    exclude_sentences=all_test_sentences
)
init_time = time.time() - init_start_time
print(f"Initialization time: {init_time:.2f}s")

print(f"\n{'='*70}")
print("CONTEXT-AWARE DISAMBIGUATION (Pure MLM-PLL)")
print(f"{'='*70}")

eval_start_time = time.time()
metrics, results = model.evaluate(test_data, show_progress=True, use_mlm=True)
eval_time = time.time() - eval_start_time

print(f"\nAmbiguous words (bote/buti): {metrics['total_ambiguous']}")
print(f"Correct disambiguations: {metrics['correct_ambiguous']}")
print(f"★ Context-aware accuracy: {metrics['ambiguous_accuracy']:.2%} ★")
print(f"Evaluation time: {eval_time:.2f}s ({(eval_time/metrics['total_ambiguous'])*1000:.2f}ms per ambiguity)")
```

## T. MLM vs Multi-Feature Comparison

```python
"""
Quick test: Pure MLM (semantic only) vs MLM + Multi-Feature
"""

def run_test(disambiguator, sentences_a, sentences_b, word_a, word_b, candidates, use_mlm, weights_override=None):
    all_sents = sentences_a + sentences_b
    test_data = []
    for sent in all_sents:
        words = get_clean_words(sent)
        ocr = []
        for w in words:
            if w == word_a or w == word_b:
                ocr.append(list(candidates))
            else:
                ocr.append(w)
        test_data.append({
            'sentence': sent, 
            'ocr_candidates': ocr, 
            'ground_truth': sent
        })
    
    metrics, details = disambiguator.evaluate(
        test_data, 
        use_mlm=use_mlm, 
        weights_override=weights_override
    )
    
    a_correct = 0
    b_correct = 0
    n_a = len(sentences_a)
    
    for i, r in enumerate(details):
        predicted = r['predicted'].lower()
        if i < n_a:
            if word_a in get_clean_words(predicted):
                a_correct += 1
        else:
            if word_b in get_clean_words(predicted):
                b_correct += 1
    
    return a_correct, len(sentences_a), b_correct, len(sentences_b)

pairs = [
    {
        'name': 'bote/buti',
        'file': 'gold_standard_dataset/sentences/03_bote_buti.txt',
        'word_a': 'bote', 'word_b': 'buti',
        'candidates': ["bote", "buti"]
    },
    {
        'name': 'itodo/ituro', 
        'file': 'gold_standard_dataset/sentences/07_itodo_ituro.txt',
        'word_a': 'itodo', 'word_b': 'ituro',
        'candidates': ["itodo", "ituro"]
    },
]

methods = [
    {
        'name': 'Pure MLM (Semantic Only)',
        'use_mlm': True,
        'weights': {'semantic': 1.0, 'frequency': 0.0, 'cooccurrence': 0.0, 'morphology': 0.0}
    },
    {
        'name': 'MLM + Multi-Feature',
        'use_mlm': True,
        'weights': None
    },
]

results_table = {}

for pair in pairs:
    print(f"\n{'─' * 60}")
    print(f"  Testing: {pair['name']}")
    print(f"{'─' * 60}")
    
    sents_a, sents_b = parse_sentence_file(pair['file'], pair['word_a'], pair['word_b'])
    results_table[pair['name']] = {}
    
    for method in methods:
        print(f"\n  {method['name']}...")
        a_c, a_t, b_c, b_t = run_test(
            disambiguator, sents_a, sents_b,
            pair['word_a'], pair['word_b'], pair['candidates'],
            use_mlm=method['use_mlm'],
            weights_override=method['weights']
        )
        total = a_c + b_c
        overall = (total / (a_t + b_t)) * 100
        print(f"    {pair['word_a']}: {a_c}/{a_t}, {pair['word_b']}: {b_c}/{b_t}, Overall: {overall:.0f}%")
        results_table[pair['name']][method['name']] = {
            'a': f"{a_c}/{a_t}", 'b': f"{b_c}/{b_t}", 'overall': overall
        }
```

## U. Comprehensive Evaluation Script

```python
#!/usr/bin/env python3
"""
Evaluation Script for Baybayin Disambiguation Model
"""

import json
import sys
import argparse
from pathlib import Path

from src.disambiguator import BaybayinDisambiguator
from src.baselines import MaBaybayDefault, EmbeddingOnly, LLMBaseline

TEST_SENTENCES_FILE = "archive/old_dataset/dataset/processed/test_sentences_500.txt"
TEST_CANDIDATES_FILE = "archive/old_dataset/dataset/processed/candidates_results_v2.json"
CORPUS_FILES = [
    "corpus/Tagalog_Literary_Text.txt",
    "corpus/Tagalog_Religious_Text.txt"
]
OUTPUT_DIR = Path("gold_standard_dataset/results")

def load_test_data():
    with open(TEST_SENTENCES_FILE, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    with open(TEST_CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    return sentences, candidates

def print_results(name: str, metrics: dict):
    print(f"\n{name}")
    print("-" * 50)
    print(f"  Total Word Accuracy:     {metrics['total_accuracy']:.2%}")
    print(f"  Ambiguous Word Accuracy: {metrics['ambiguous_accuracy']:.2%}")
    print(f"  Correct Ambiguous:       {metrics['correct_ambiguous']}/{metrics['total_ambiguous']}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Baybayin Disambiguation Model')
    parser.add_argument('--baselines', action='store_true', help='Include baseline comparisons')
    parser.add_argument('--llm', type=str, choices=['gemini', 'openai', 'ollama'],
                       help='Test with LLM')
    parser.add_argument('--llm-limit', type=int, default=None,
                       help='Limit number of sentences for LLM')
    parser.add_argument('--weights', nargs=4, type=float, 
                       metavar=('SEM', 'FREQ', 'COOC', 'MORPH'),
                       help='Feature weights')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BAYBAYIN DISAMBIGUATION - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    print("\nLoading test data...")
    test_sentences, test_data = load_test_data()
    print(f"  [OK] {len(test_sentences)} test sentences")
    print(f"  [OK] {len(test_data)} candidate entries")
    
    results_all = {}
    
    if args.baselines:
        print("\n" + "=" * 70)
        print("BASELINE MODELS")
        print("=" * 70)
        
        default_model = MaBaybayDefault()
        metrics_default, _ = default_model.evaluate(test_data)
        results_all['mabaybay_default'] = metrics_default
        print_results("MaBaybay Default (First Candidate)", metrics_default)
        
        emb_model = EmbeddingOnly()
        metrics_emb, _ = emb_model.evaluate(test_data)
        results_all['embedding_only'] = metrics_emb
        print_results("Embedding-Only (WE-Only)", metrics_emb)
    
    if args.llm:
        print("\n" + "=" * 70)
        print(f"LLM BASELINE ({args.llm.upper()})")
        print("=" * 70)
        
        try:
            llm_model = LLMBaseline(provider=args.llm)
            llm_data = test_data[:args.llm_limit] if args.llm_limit else test_data
            if args.llm_limit:
                print(f"  [INFO] Limiting to {args.llm_limit} sentences")
            
            metrics_llm, _ = llm_model.evaluate(llm_data)
            results_all['llm'] = metrics_llm
            print_results(f"LLM ({args.llm} - {llm_model.model})", metrics_llm)
        except Exception as e:
            print(f"  [ERROR] Failed to run LLM baseline: {e}")
    
    print("\n" + "=" * 70)
    print("GRAPH-BASED MODEL (MLM + Multi-Feature) - CLEAN EVALUATION")
    print("=" * 70)
    
    if args.weights:
        weights = {
            'semantic': args.weights[0],
            'frequency': args.weights[1],
            'cooccurrence': args.weights[2],
            'morphology': args.weights[3]
        }
    else:
        weights = None
    
    model = BaybayinDisambiguator(
        corpus_files=CORPUS_FILES,
        exclude_sentences=test_sentences,
        weights=weights
    )
    
    metrics, detailed_results = model.evaluate(test_data)
    results_all['graph_based'] = metrics
    print_results("Graph-Based + Features (Clean Evaluation)", metrics)
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print()
    print(f"{'Method':<45} | {'Ambiguous Accuracy':>18}")
    print("-" * 70)
    
    if args.baselines:
        print(f"{'MaBaybay Default (no disambiguation)':<45} | {results_all['mabaybay_default']['ambiguous_accuracy']:>17.2%}")
        print(f"{'Embedding-Only (WE-Only)':<45} | {results_all['embedding_only']['ambiguous_accuracy']:>17.2%}")
    
    if 'llm' in results_all:
        print(f"{'LLM (' + args.llm + ') - Our Test':<45} | {results_all['llm']['ambiguous_accuracy']:>17.2%}")
    
    print(f"{'bAI-bAI WE-Only (reported in paper)':<45} | {'77.46%':>18}")
    print(f"{'bAI-bAI LLM (reported in paper)':<45} | {'90.52%':>18}")
    print(f"{'Our Graph+Features (Clean Evaluation)':<45} | {metrics['ambiguous_accuracy']:>17.2%}")
    
    improvement_we = metrics['ambiguous_accuracy'] - 0.7746
    improvement_llm = metrics['ambiguous_accuracy'] - 0.9052
    print()
    print(f"Improvement over WE-Only:  {improvement_we:+.2%}")
    print(f"Improvement over LLM:      {improvement_llm:+.2%}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "evaluation_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'test_sentences': len(test_sentences),
                'weights': model.weights
            },
            'results': {k: {kk: float(vv) if isinstance(vv, float) else vv 
                          for kk, vv in v.items()} 
                       for k, v in results_all.items()}
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_file}")

if __name__ == "__main__":
    main()
```

## V. Image Generation

```python
"""
Generate Baybayin script images from Filipino sentences.
"""

from html2image import Html2Image
from PIL import Image
import os
import csv
import base64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, "tagalog stylized.ttf")
FONT_SIZE = 150
FONT_WEIGHT = "normal"
LINE_HEIGHT = 1.2
WORDS_PER_ROW = 3.0
NORMAL_WORD_SPACING = 1.0
EXTRA_WORD_SPACING = 2.5
IMAGE_PADDING = 30
IMAGE_PADDING_TOP = 50
IMAGE_PADDING_BOTTOM = 50

SENTENCES_FILE = os.path.join(SCRIPT_DIR, "filipino_sentences.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Baybayin_Sample_Images")
ANNOTATIONS_FILE = os.path.join(SCRIPT_DIR, "annotations.csv")

def latin_to_baybayin_tagalog_stylized(text):
    import re
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[.,!?;:"""''`(){}\[\]<>—–-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    vowels = set('aeiou')
    consonants = set('bkdghlmnprstwy')
    
    def process_word(word):
        result = []
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char == 'n' and i + 1 < len(word) and word[i + 1] == 'g':
                if i + 2 < len(word) and word[i + 2] in vowels:
                    vowel = word[i + 2]
                    if vowel == 'a':
                        result.append('N')
                    elif vowel in 'ie':
                        result.append('Ni')
                    elif vowel in 'ou':
                        result.append('Nu')
                    i += 3
                else:
                    result.append('N+')
                    i += 2
                    
            elif char in consonants:
                typing_char = 'd' if char == 'r' else char
                
                if i + 1 < len(word) and word[i + 1] in vowels:
                    vowel = word[i + 1]
                    if vowel == 'a':
                        result.append(typing_char)
                    elif vowel in 'ie':
                        result.append(typing_char + 'i')
                    elif vowel in 'ou':
                        result.append(typing_char + 'u')
                    i += 2
                else:
                    result.append(typing_char + '+')
                    i += 1
                    
            elif char in vowels:
                if char == 'a':
                    result.append('A')
                elif char in 'ie':
                    result.append('I')
                elif char in 'ou':
                    result.append('U')
                i += 1
            else:
                i += 1
        
        return ''.join(result)
    
    words = text.split(' ')
    processed_words = [process_word(word) for word in words]
    
    result_parts = []
    for i, baybayin_word in enumerate(processed_words):
        result_parts.append(baybayin_word)
        if i < len(processed_words) - 1:
            result_parts.append(' ')
    
    return ''.join(result_parts)

def generate_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(FONT_PATH):
        print(f"ERROR: Font file '{FONT_PATH}' not found.")
        return
    
    hti = Html2Image(output_path=OUTPUT_DIR)
    
    print(f"Using font: {FONT_PATH}")
    print("Starting image generation with Tagalog Stylized font...")
    
    annotations = []
    
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            sentence = line.strip()
            if not sentence:
                continue
            
            baybayin_text = latin_to_baybayin_tagalog_stylized(sentence)
            
            print(f"Processing sentence {i+1}: {sentence[:50]}...")
            
            with open(FONT_PATH, 'rb') as font_file:
                font_base64 = base64.b64encode(font_file.read()).decode('utf-8')
            
            unique_font_name = f'TagalogStylized_{i}'
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    @font-face {{
                        font-family: '{unique_font_name}';
                        src: url(data:font/truetype;charset=utf-8;base64,{font_base64}) format('truetype');
                        font-display: block;
                    }}
                    html, body {{
                        margin: 0;
                        padding: {IMAGE_PADDING}px;
                        background-color: white !important;
                    }}
                    .baybayin {{
                        font-family: '{unique_font_name}' !important;
                        font-size: {FONT_SIZE}px;
                        font-weight: {FONT_WEIGHT};
                        line-height: {LINE_HEIGHT};
                        color: black;
                        white-space: pre-wrap;
                        background-color: white;
                        display: inline-block;
                        text-rendering: optimizeLegibility;
                        letter-spacing: 2px;
                    }}
                </style>
            </head>
            <body>
                <div class="baybayin">{baybayin_text}</div>
            </body>
            </html>
            """
            
            image_filename = f"sentence_{i+1}.png"
            hti.screenshot(
                html_str=html_content,
                save_as=image_filename,
                size=(3000, 2000)
            )
            
            img_path = os.path.join(OUTPUT_DIR, image_filename)
            img = Image.open(img_path).convert('RGBA')
            
            white_bg = Image.new('RGB', img.size, 'white')
            white_bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            
            gray = white_bg.convert('L')
            bbox = gray.getbbox()
            if bbox:
                cropped = white_bg.crop(bbox)
                padded = Image.new('RGB', 
                    (cropped.width + 40, cropped.height + 40), 'white')
                padded.paste(cropped, (20, 20))
                padded.save(img_path, 'PNG', quality=95)
            
            annotations.append({
                'image': image_filename,
                'latin': sentence,
                'baybayin_font': baybayin_text
            })
            
            print(f"  ✓ Saved: {image_filename}")
    
    with open(ANNOTATIONS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'latin', 'baybayin_font'])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"\n✓ Generated {len(annotations)} images")
    print(f"✓ Annotations saved to: {ANNOTATIONS_FILE}")

if __name__ == "__main__":
    generate_images()
```

---

End of Source Code Appendix
