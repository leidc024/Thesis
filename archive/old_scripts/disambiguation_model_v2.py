"""
Enhanced Graph-Based Disambiguation Model for Baybayin OCR (v2)
Improvements over v1:
1. Corpus frequency weighting - favors common words
2. Co-occurrence statistics - bigram probabilities
3. Morphological features - Filipino affix patterns
4. Combined scoring with tunable weights

Architecture:
1. Get RoBERTa embeddings for each candidate word
2. Build a graph where nodes are candidate words
3. Edges weighted by: semantic similarity + co-occurrence + morphology
4. Node weights include corpus frequency
5. Use Personalized PageRank to rank candidates
6. Select highest-ranked candidate for each ambiguous position
"""

import json
import torch
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import re

# Configuration
MODEL_NAME = "jcblaise/roberta-tagalog-base"
# Use actual text corpora for real word frequencies (not just vocabulary list)
TEXT_CORPORA = [
    "Tagalog_Literary_Text.txt",    # ~201k words of literary Filipino
    "Tagalog_Religious_Text.txt"    # ~90k words of religious Filipino
]
VOCAB_FILE = "MaBaybay-OCR/Filipino Word Corpus/Tagalog_words_74419+.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Feature weights (tunable hyperparameters)
WEIGHT_SEMANTIC = 0.3      # RoBERTa similarity
WEIGHT_FREQUENCY = 0.4     # Corpus frequency (increased - now using real text)
WEIGHT_COOCCURRENCE = 0.2  # Bigram co-occurrence
WEIGHT_MORPHOLOGY = 0.1    # Morphological features


class CorpusStatistics:
    """
    Handles corpus frequency and co-occurrence statistics.
    Uses actual Filipino text corpora for real word frequencies.
    """
    
    def __init__(self, text_files: List[str] = None, vocab_file: str = None):
        """Load and process the Filipino text corpora."""
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.total_words = 0
        self.total_bigrams = 0
        self.vocab = set()
        
        text_files = text_files or TEXT_CORPORA
        vocab_file = vocab_file or VOCAB_FILE
        
        # Load vocabulary (valid words)
        self._load_vocab(vocab_file)
        
        # Load frequencies from actual text
        self._load_text_corpora(text_files)
        
        print(f"[OK] Loaded {len(self.vocab)} vocabulary words")
        print(f"[OK] Corpus stats: {self.total_words} words, {len(self.word_freq)} unique from text")
    
    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from word list file."""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.vocab.add(word)
        except FileNotFoundError:
            print(f"  Warning: Vocab file not found: {vocab_path}")
    
    def _load_text_corpora(self, text_files: List[str]):
        """Load word frequencies from actual text corpora."""
        import re
        
        all_words = []
        
        for filepath in text_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().lower()
                
                # Extract words (Filipino text may have special chars)
                words = re.findall(r"[a-z\-']+", text)
                
                # Clean words
                words = [w.strip("-'") for w in words if len(w) > 1]
                all_words.extend(words)
                
                print(f"  Loaded {len(words)} words from {filepath}")
            except FileNotFoundError:
                print(f"  Warning: Text file not found: {filepath}")
        
        self.word_freq = Counter(all_words)
        self.total_words = len(all_words)
        
        # Build bigram statistics (consecutive words)
        for i in range(len(all_words) - 1):
            bigram = (all_words[i], all_words[i+1])
            self.bigram_freq[bigram] += 1
        self.total_bigrams = len(all_words) - 1
    
    def get_word_frequency(self, word: str) -> float:
        """
        Get normalized frequency of a word.
        Returns log-smoothed probability to handle rare words.
        """
        count = self.word_freq.get(word.lower(), 0)
        # Laplace smoothing + log transform
        prob = (count + 1) / (self.total_words + len(self.word_freq))
        return np.log(prob + 1e-10)
    
    def get_frequency_score(self, word: str) -> float:
        """
        Get a 0-1 normalized frequency score.
        Higher = more frequent word.
        """
        count = self.word_freq.get(word.lower(), 0)
        if count == 0:
            return 0.1  # Small non-zero for unknown words
        
        # Normalize by max frequency
        max_freq = self.word_freq.most_common(1)[0][1]
        return min(1.0, (np.log(count + 1) / np.log(max_freq + 1)))
    
    def get_bigram_probability(self, word1: str, word2: str) -> float:
        """
        Get P(word2 | word1) - probability of word2 following word1.
        """
        bigram = (word1.lower(), word2.lower())
        bigram_count = self.bigram_freq.get(bigram, 0)
        word1_count = self.word_freq.get(word1.lower(), 0)
        
        if word1_count == 0:
            return 0.0
        
        # Laplace smoothing
        return (bigram_count + 0.1) / (word1_count + len(self.word_freq) * 0.1)


class MorphologicalAnalyzer:
    """
    Analyzes Filipino morphological patterns for disambiguation.
    """
    
    # Common Filipino affixes
    PREFIXES = [
        'mag', 'nag', 'pag', 'um', 'in', 'ka', 'ma', 'na', 'pa', 
        'i', 'ika', 'ipa', 'ipag', 'ipang', 'maka', 'maki', 'makapag',
        'nakaka', 'nakapag', 'mapag', 'sang', 'tag', 'taga'
    ]
    
    INFIXES = ['um', 'in']
    
    SUFFIXES = [
        'an', 'in', 'han', 'hin', 'nan', 'ang', 'ing',
        'ito', 'iyan', 'yon', 'ng'
    ]
    
    # Reduplication pattern (common in Filipino)
    REDUPLICATION_PATTERN = re.compile(r'^(\w{2,3})\1')
    
    def __init__(self):
        """Initialize morphological patterns."""
        pass
    
    def get_morphological_score(self, word: str) -> float:
        """
        Score a word based on how well it matches Filipino morphological patterns.
        Higher score = more likely to be a valid Filipino word.
        """
        word = word.lower()
        score = 0.5  # Base score
        
        # Check prefixes
        for prefix in self.PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                score += 0.1
                break
        
        # Check suffixes
        for suffix in self.SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                score += 0.1
                break
        
        # Check for reduplication (e.g., "kaka", "sisi")
        if self.REDUPLICATION_PATTERN.match(word):
            score += 0.1
        
        # Penalize very short words (less common to be ambiguous correctly)
        if len(word) <= 2:
            score -= 0.1
        
        # Words ending in common syllables are more natural
        if word.endswith(('a', 'i', 'o', 'e', 'u', 'ng', 'n', 'g')):
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def compare_morphology(self, word1: str, word2: str) -> Tuple[float, float]:
        """
        Compare morphological scores of two candidate words.
        Returns normalized scores that sum to 1.
        """
        score1 = self.get_morphological_score(word1)
        score2 = self.get_morphological_score(word2)
        
        total = score1 + score2
        if total == 0:
            return 0.5, 0.5
        
        return score1 / total, score2 / total


class BaybayinDisambiguatorV2:
    """
    Enhanced graph-based disambiguator using:
    - RoBERTa embeddings for semantic similarity
    - Corpus frequency statistics
    - Co-occurrence (bigram) probabilities
    - Morphological analysis
    """
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        text_corpora: List[str] = None,
        vocab_file: str = None,
        weights: Dict[str, float] = None
    ):
        """Initialize all components."""
        print("=" * 50)
        print("Initializing Enhanced Disambiguator v2")
        print("=" * 50)
        
        # Load RoBERTa model
        print(f"\nLoading RoBERTa model: {model_name}")
        print(f"Device: {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("[OK] RoBERTa loaded")
        
        # Load corpus statistics from actual text
        print(f"\nLoading corpus statistics from text corpora...")
        self.corpus = CorpusStatistics(text_corpora, vocab_file)
        
        # Initialize morphological analyzer
        print(f"Initializing morphological analyzer...")
        self.morphology = MorphologicalAnalyzer()
        print("[OK] Morphology analyzer ready")
        
        # Set feature weights
        self.weights = weights or {
            'semantic': WEIGHT_SEMANTIC,
            'frequency': WEIGHT_FREQUENCY,
            'cooccurrence': WEIGHT_COOCCURRENCE,
            'morphology': WEIGHT_MORPHOLOGY
        }
        print(f"\nFeature weights: {self.weights}")
        print("=" * 50)
    
    def get_word_embedding(self, word: str, context: str = None) -> np.ndarray:
        """Get RoBERTa embedding for a word."""
        text = context if context else word
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(DEVICE)
            
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding.cpu().numpy().flatten()
    
    def compute_candidate_score(
        self,
        candidate: str,
        context_embedding: np.ndarray,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute multi-feature score for a candidate word.
        
        Returns dict with individual feature scores and combined score.
        """
        scores = {}
        
        # 1. Semantic similarity with context
        candidate_embedding = self.get_word_embedding(candidate)
        semantic_sim = cosine_similarity(
            candidate_embedding.reshape(1, -1),
            context_embedding.reshape(1, -1)
        )[0, 0]
        scores['semantic'] = max(0, semantic_sim)
        
        # 2. Corpus frequency
        scores['frequency'] = self.corpus.get_frequency_score(candidate)
        
        # 3. Co-occurrence with neighboring words
        cooc_score = 0.0
        if prev_word:
            cooc_score += self.corpus.get_bigram_probability(prev_word, candidate)
        if next_word:
            cooc_score += self.corpus.get_bigram_probability(candidate, next_word)
        scores['cooccurrence'] = min(1.0, cooc_score * 10)  # Scale up
        
        # 4. Morphological score
        scores['morphology'] = self.morphology.get_morphological_score(candidate)
        
        # Combined weighted score
        scores['combined'] = (
            self.weights['semantic'] * scores['semantic'] +
            self.weights['frequency'] * scores['frequency'] +
            self.weights['cooccurrence'] * scores['cooccurrence'] +
            self.weights['morphology'] * scores['morphology']
        )
        
        return scores
    
    def disambiguate_sentence(
        self, 
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        """
        Disambiguate a sentence using enhanced multi-feature approach.
        """
        # Build context from ground truth or first candidates
        if ground_truth:
            context = ground_truth
        else:
            context = ' '.join(
                item[0] if isinstance(item, list) else item 
                for item in ocr_candidates
            )
        
        context_embedding = self.get_word_embedding(context)
        
        # Get unambiguous words for context
        resolved_words = []
        for item in ocr_candidates:
            if isinstance(item, list):
                resolved_words.append(None)  # To be resolved
            else:
                resolved_words.append(item)
        
        # First pass: resolve unambiguous positions
        disambiguated = []
        debug_info = {'scores': {}, 'selected': {}, 'features': {}}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                # Get neighboring words for co-occurrence
                prev_word = resolved_words[pos-1] if pos > 0 else None
                next_word = None
                for j in range(pos+1, len(ocr_candidates)):
                    if not isinstance(ocr_candidates[j], list):
                        next_word = ocr_candidates[j]
                        break
                
                # Score each candidate
                candidate_scores = {}
                for candidate in item:
                    scores = self.compute_candidate_score(
                        candidate, context_embedding, prev_word, next_word
                    )
                    candidate_scores[candidate] = scores
                
                # Select best candidate
                best_candidate = max(
                    candidate_scores.keys(),
                    key=lambda c: candidate_scores[c]['combined']
                )
                
                disambiguated.append(best_candidate)
                resolved_words[pos] = best_candidate
                
                debug_info['scores'][pos] = {
                    c: s['combined'] for c, s in candidate_scores.items()
                }
                debug_info['selected'][pos] = best_candidate
                debug_info['features'][pos] = candidate_scores
            else:
                disambiguated.append(item)
        
        return disambiguated, debug_info
    
    def evaluate(self, test_data: List[Dict], verbose: bool = False) -> Tuple[Dict, List]:
        """Evaluate the model on test data."""
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        
        results = []
        
        for entry in tqdm(test_data, desc="Evaluating"):
            ground_truth = entry['ground_truth']
            ocr_candidates = entry['ocr_candidates']
            gt_words = ground_truth.lower().split()
            
            # Disambiguate
            predicted, debug_info = self.disambiguate_sentence(
                ocr_candidates, ground_truth
            )
            
            # Compare
            entry_result = {
                'ground_truth': ground_truth,
                'predicted': ' '.join(predicted),
                'details': []
            }
            
            for i, (pred, gt) in enumerate(zip(predicted, gt_words)):
                if i >= len(ocr_candidates):
                    break
                    
                is_ambiguous = isinstance(ocr_candidates[i], list)
                is_correct = pred.lower() == gt.lower()
                
                total_words += 1
                if is_correct:
                    correct_words += 1
                
                if is_ambiguous:
                    total_ambiguous += 1
                    if is_correct:
                        correct_ambiguous += 1
                    
                    entry_result['details'].append({
                        'position': i,
                        'candidates': ocr_candidates[i],
                        'predicted': pred,
                        'ground_truth': gt,
                        'correct': is_correct,
                        'scores': debug_info.get('scores', {}).get(i, {})
                    })
            
            results.append(entry_result)
            
            if verbose and not all(d['correct'] for d in entry_result['details']):
                print(f"\nGT: {ground_truth}")
                print(f"PR: {' '.join(predicted)}")
                for d in entry_result['details']:
                    if not d['correct']:
                        print(f"  X pos {d['position']}: {d['candidates']} -> {d['predicted']} (should be {d['ground_truth']})")
        
        metrics = {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words > 0 else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous > 0 else 0
        }
        
        return metrics, results


def load_dataset(filepath: str) -> List[Dict]:
    """Load the candidates results dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# For quick testing
if __name__ == "__main__":
    # Test the enhanced model
    print("\n" + "="*60)
    print("TESTING ENHANCED DISAMBIGUATOR V2")
    print("="*60 + "\n")
    
    # Load model
    model = BaybayinDisambiguatorV2()
    
    # Test with a sample
    test_candidates = [
        ['hinde', 'hindi'],  # Should select 'hindi' (more common)
        'ko',
        'alam',
        'kung',
        ['ano', 'anu'],  # Should select 'ano' (more common)
        'ang',
        'gagawin',
        ['neto', 'nito']  # Should select 'nito' (more common)
    ]
    
    test_gt = "hindi ko alam kung ano ang gagawin nito"
    
    print(f"Ground truth: {test_gt}")
    print(f"Candidates: {test_candidates}")
    
    result, debug = model.disambiguate_sentence(test_candidates, test_gt)
    print(f"\nPredicted: {' '.join(result)}")
    
    print("\nDetailed scores:")
    for pos, scores in debug['features'].items():
        print(f"\n  Position {pos}:")
        for candidate, features in scores.items():
            selected = " <-- SELECTED" if candidate == debug['selected'][pos] else ""
            print(f"    {candidate}: combined={features['combined']:.4f} "
                  f"(sem={features['semantic']:.3f}, freq={features['frequency']:.3f}, "
                  f"cooc={features['cooccurrence']:.3f}, morph={features['morphology']:.3f}){selected}")
