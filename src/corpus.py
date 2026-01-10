"""
Corpus Statistics Module
Handles word frequency and co-occurrence statistics from Filipino text corpora.
"""

import re
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Set


class CorpusStatistics:
    """
    Computes and manages corpus statistics for disambiguation.
    
    Features:
    - Word frequency from Filipino text corpora
    - Bigram (co-occurrence) probabilities
    - Optional exclusion of test sentences to prevent data leakage
    """
    
    def __init__(
        self, 
        text_files: List[str],
        exclude_sentences: List[str] = None,
        vocab_file: str = None
    ):
        """
        Initialize corpus statistics.
        
        Args:
            text_files: List of paths to Filipino text corpora
            exclude_sentences: Sentences to exclude (for clean evaluation)
            vocab_file: Optional vocabulary file for word validation
        """
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.total_words = 0
        self.total_bigrams = 0
        self.vocab: Set[str] = set()
        self.excluded_count = 0
        
        # Normalize excluded sentences
        if exclude_sentences:
            self.excluded = set(
                self._normalize_sentence(s) for s in exclude_sentences
            )
        else:
            self.excluded = set()
        
        # Load vocabulary if provided
        if vocab_file:
            self._load_vocab(vocab_file)
        
        # Load corpus statistics
        self._load_corpora(text_files)
    
    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for matching."""
        sentence = sentence.lower().strip()
        sentence = sentence.replace('-', ' ')
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence
    
    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from word list file."""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.vocab.add(word)
            print(f"  [OK] Vocabulary: {len(self.vocab)} words")
        except FileNotFoundError:
            print(f"  [!] Vocab file not found: {vocab_path}")
    
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
    
    def get_bigram_probability(self, word1: str, word2: str) -> float:
        """
        Get P(word2 | word1) - probability of word2 following word1.
        Uses Laplace smoothing.
        """
        bigram = (word1.lower(), word2.lower())
        bigram_count = self.bigram_freq.get(bigram, 0)
        word1_count = self.word_freq.get(word1.lower(), 0)
        
        if word1_count == 0:
            return 0.0
        
        # Laplace smoothing
        vocab_size = len(self.word_freq)
        return (bigram_count + 0.1) / (word1_count + vocab_size * 0.1)
    
    def is_valid_word(self, word: str) -> bool:
        """Check if word is in vocabulary or corpus."""
        word_lower = word.lower()
        return word_lower in self.vocab or word_lower in self.word_freq
