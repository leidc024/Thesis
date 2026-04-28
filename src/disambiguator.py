"""
Baybayin Disambiguation Model
Context-aware transliteration disambiguation using RoBERTa MLM scoring and linguistic features.

This module implements a multi-feature approach combining:
1. Semantic context (RoBERTa Masked Language Model pseudo-log-likelihood)
2. Corpus frequency statistics
3. Co-occurrence (bigram) probabilities  
4. Morphological analysis

Architecture:
- Input: OCR candidates (ambiguous positions have multiple options)
- Process: Score each candidate using weighted multi-feature approach
- Output: Disambiguated sentence with best candidates selected

Key innovation: Uses Pseudo-Log-Likelihood (PLL) scoring from the MLM head
instead of cosine similarity of mean-pooled embeddings. PLL directly measures
how well each candidate fits the sentence context, providing much stronger
semantic signal especially for rare words.
"""

import math
import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
try:
    from tqdm.auto import tqdm as tqdm_auto
    tqdm_auto.disable = True  # Disable tqdm.auto used by transformers
except:
    pass

# Disable tqdm by default (enable only for evaluation mode with show_progress=True)
tqdm.disable = True

from .corpus import CorpusStatistics
from .morphology import MorphologicalAnalyzer


# Default configuration
DEFAULT_MODEL = "jcblaise/roberta-tagalog-base"
# Weights for ambiguous words: MLM-PLL only (context-aware)
DEFAULT_WEIGHTS = {
    'semantic': 1.0,      # MLM PLL scoring (context-aware, the primary signal)
    'frequency': 0.0,     # Not used for ambiguous words with context
    'cooccurrence': 0.0,  # Not used for ambiguous words with context
    'morphology': 0.0     # Not used for ambiguous words with context
}
# For unambiguous words: frequency-only scoring
FREQUENCY_ONLY_WEIGHTS = {
    'semantic': 0.0,
    'frequency': 1.0,
    'cooccurrence': 0.0,
    'morphology': 0.0
}

# Text corpora paths
DEFAULT_CORPORA = [
    "corpus/Tagalog_Literary_Text.txt",
    "corpus/Tagalog_Religious_Text.txt",
    "corpus/Tagalog_Balita_Texts_Balanced.txt"
]


class BaybayinDisambiguator:
    """
    Context-aware Baybayin transliteration disambiguator.
    
    Uses a multi-feature scoring approach to select the most likely
    candidate for ambiguous Baybayin-to-Filipino transliterations.
    
    Ambiguity types handled:
    - E/I confusion (ᜁ can be 'e' or 'i')
    - O/U confusion (ᜂ can be 'o' or 'u')
    - D/R confusion (ᜇ can be 'd' or 'r')
    
    Example:
        >>> model = BaybayinDisambiguator(corpus_files=["corpus.txt"])
        >>> candidates = ["ang", ["dito", "rito"], "ay", ["sino", "seno"]]
        >>> result, debug = model.disambiguate(candidates)
        >>> print(result)  # ['ang', 'dito', 'ay', 'sino']
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        corpus_files: List[str] = None,
        exclude_sentences: List[str] = None,
        weights: Dict[str, float] = None,
        device: str = None
    ):
        """
        Initialize the disambiguator.
        
        Args:
            model_name: HuggingFace model identifier for embeddings
            corpus_files: List of paths to Filipino text corpora
            exclude_sentences: Sentences to exclude from corpus (for evaluation)
            weights: Feature weights dict (semantic, frequency, cooccurrence, morphology)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights or DEFAULT_WEIGHTS
        corpus_files = corpus_files or DEFAULT_CORPORA
        
        print("=" * 60)
        print("BAYBAYIN DISAMBIGUATOR - Initialization")
        print("=" * 60)
        
        # Load RoBERTa model (MLM head for pseudo-log-likelihood scoring)
        print(f"\n[1/3] Loading RoBERTa MLM: {model_name}")
        print(f"      Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("      [OK] Model loaded (with MLM head for PLL scoring)")
        
        # Load corpus statistics
        print(f"\n[2/3] Loading corpus statistics...")
        self.corpus = CorpusStatistics(
            text_files=corpus_files,
            exclude_sentences=exclude_sentences
        )
        
        # Initialize morphological analyzer
        print(f"\n[3/3] Initializing morphological analyzer...")
        self.morphology = MorphologicalAnalyzer()
        print("      [OK] Ready")
        
        print(f"\nFeature weights: {self.weights}")
        print("=" * 60 + "\n")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get mean-pooled RoBERTa embedding for text.
        Uses hidden states from the MLM model's base encoder.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (hidden_size,)
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Use output_hidden_states to get base model embeddings from MLM model
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]  # Last layer hidden states
            
            # Mean pooling with attention mask
            mask = inputs['attention_mask'].unsqueeze(-1)
            mask = mask.expand(embeddings.size()).float()
            sum_emb = torch.sum(embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_emb = sum_emb / sum_mask
            
            return mean_emb.cpu().numpy().flatten()
    
    def _find_subtoken_positions(
        self,
        token_ids: torch.Tensor,
        subtoken_ids: List[int]
    ) -> List[int]:
        """
        Find positions of a subtoken sequence within a full token sequence.
        
        Args:
            token_ids: Full tokenized sentence (tensor)
            subtoken_ids: Subtoken IDs of the candidate word
            
        Returns:
            List of positions where the subtoken sequence starts, or empty list
        """
        seq = token_ids.tolist()
        sub = subtoken_ids
        n = len(sub)
        
        for i in range(len(seq) - n + 1):
            if seq[i:i+n] == sub:
                return list(range(i, i + n))
        
        return []
    
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
    
    def score_candidate(
        self,
        candidate: str,
        context_embedding: np.ndarray,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None,
        mlm_score: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute multi-feature score for a candidate word.
        
        Args:
            candidate: Candidate word to score
            context_embedding: Embedding of sentence context (fallback for semantic)
            prev_word: Previous word in sentence (for bigram)
            next_word: Next word in sentence (for bigram)
            mlm_score: Pre-computed MLM score from PLL (if available)
            
        Returns:
            Dict with individual feature scores and combined score
        """
        scores = {}
        
        # 1. Semantic similarity with context
        if mlm_score is not None:
            # Use MLM-based PLL score (much stronger signal than cosine similarity)
            scores['semantic'] = mlm_score
        else:
            # Fallback to cosine similarity of mean-pooled embeddings
            cand_emb = self.get_embedding(candidate)
            semantic_sim = cosine_similarity(
                cand_emb.reshape(1, -1),
                context_embedding.reshape(1, -1)
            )[0, 0]
            scores['semantic'] = max(0.0, float(semantic_sim))
        
        # 2. Corpus frequency
        scores['frequency'] = self.corpus.get_frequency_score(candidate)
        
        # 3. Co-occurrence (bigram probability)
        cooc = 0.0
        if prev_word:
            cooc += self.corpus.get_bigram_probability(prev_word, candidate)
        if next_word:
            cooc += self.corpus.get_bigram_probability(candidate, next_word)
        scores['cooccurrence'] = min(1.0, cooc * 10)  # Scale up
        
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
        
        # Build sentence word list for MLM scoring
        # Unambiguous words stay as-is; ambiguous positions are placeholders
        sentence_words = []
        for c in ocr_candidates:
            if isinstance(c, str):
                sentence_words.append(c)
            else:
                sentence_words.append(None)  # Placeholder for ambiguous position
        
        # Track resolved words for co-occurrence
        resolved = [
            None if isinstance(c, list) else c 
            for c in ocr_candidates
        ]
        
        result = []
        debug = {'selected': {}, 'scores': {}}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                # Get neighboring words for co-occurrence
                prev_word = resolved[pos - 1] if pos > 0 else None
                next_word = None
                for j in range(pos + 1, len(ocr_candidates)):
                    if not isinstance(ocr_candidates[j], list):
                        next_word = ocr_candidates[j]
                        break
                
                if use_mlm:
                    # Build MLM sentence: fill in resolved words for other ambiguous positions
                    mlm_words = list(sentence_words)
                    for i in range(len(mlm_words)):
                        if mlm_words[i] is None and i != pos:
                            # Use resolved word if available, else first candidate
                            if resolved[i]:
                                mlm_words[i] = resolved[i]
                            elif isinstance(ocr_candidates[i], list):
                                mlm_words[i] = ocr_candidates[i][0]
                    
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
                else:
                    # Use cosine similarity for semantic scoring (no MLM)
                    scores = {
                        c: self.score_candidate(
                            c, context_embedding, prev_word, next_word
                        )
                        for c in item
                    }
                
                # Select best candidate
                best = max(scores.keys(), key=lambda c: scores[c]['combined'])
                
                result.append(best)
                resolved[pos] = best
                debug['selected'][pos] = best
                debug['scores'][pos] = {c: s['combined'] for c, s in scores.items()}
            else:
                # For unambiguous words: use frequency score only (fallback method)
                freq_score = self.corpus.get_frequency_score(item)
                result.append(item)
        
        return result, debug
    
    def evaluate(
        self,
        test_data: List[Dict],
        show_progress: bool = True,
        use_ground_truth_context: bool = False,  # Set to False for realistic evaluation
        use_mlm: bool = True,
        weights_override: Dict[str, float] = None
    ) -> Tuple[Dict, List]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data: List of dicts with 'ground_truth' and 'ocr_candidates'
            show_progress: Show tqdm progress bar
            use_ground_truth_context: If True, use ground truth for context (unrealistic).
                                      If False, use only unambiguous words (realistic).
            use_mlm: If True, use MLM PLL for semantic. If False, use cosine similarity.
            weights_override: Temporary weight dict to use instead of self.weights.
            
        Returns:
            Tuple of (metrics_dict, detailed_results)
        """
        # Temporarily override weights if provided
        original_weights = None
        if weights_override is not None:
            original_weights = self.weights.copy()
            self.weights = weights_override
        
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        # Enable tqdm only if show_progress is True (disable globally by default for MATLAB)
        if show_progress:
            tqdm.disable = False
            iterator = tqdm(test_data, desc="Evaluating")
        else:
            tqdm.disable = True
            iterator = test_data
        
        for entry in iterator:
            gt = entry['ground_truth']
            candidates = entry['ocr_candidates']
            gt_words = gt.lower().split()
            
            # Only pass ground truth if explicitly requested (not realistic)
            if use_ground_truth_context:
                predicted, debug = self.disambiguate(candidates, gt, use_mlm=use_mlm)
            else:
                # Realistic evaluation: no ground truth, just like real MaBaybay usage
                predicted, debug = self.disambiguate(candidates, use_mlm=use_mlm)
            
            for i, (pred, gt_word) in enumerate(zip(predicted, gt_words)):
                if i >= len(candidates):
                    break
                
                is_ambiguous = isinstance(candidates[i], list)
                # Strip punctuation for fair comparison
                pred_clean = re.sub(r'[^\w]', '', pred.lower())
                gt_clean = re.sub(r'[^\w]', '', gt_word.lower())
                is_correct = pred_clean == gt_clean
                
                total_words += 1
                if is_correct:
                    correct_words += 1
                
                if is_ambiguous:
                    total_ambiguous += 1
                    if is_correct:
                        correct_ambiguous += 1
            
            results.append({
                'ground_truth': gt,
                'predicted': ' '.join(predicted),
                'debug': debug
            })
        
        # Restore original weights if they were overridden
        if original_weights is not None:
            self.weights = original_weights
        
        metrics = {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous else 0
        }
        
        return metrics, results
