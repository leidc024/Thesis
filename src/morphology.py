"""
Morphological Analyzer Module
Analyzes Filipino morphological patterns for disambiguation scoring.
"""

import re
from typing import Tuple


class MorphologicalAnalyzer:
    """
    Analyzes Filipino morphological patterns.
    
    Filipino words follow predictable patterns with:
    - Prefixes (mag-, nag-, pag-, um-, in-, etc.)
    - Infixes (-um-, -in-)
    - Suffixes (-an, -in, -han, etc.)
    - Reduplication patterns
    """
    
    # Common Filipino prefixes (ordered by length for matching)
    PREFIXES = [
        # Complex prefixes (check first)
        'nakapag', 'makapag', 'nakaka', 'mapag', 'ipang', 'ipag', 'maki',
        'maka', 'taga', 'sang', 'tag',
        # Simple prefixes
        'mag', 'nag', 'pag', 'ika', 'ipa',
        'um', 'in', 'ka', 'ma', 'na', 'pa', 'i'
    ]
    
    # Common Filipino infixes
    INFIXES = ['um', 'in']
    
    # Common Filipino suffixes
    SUFFIXES = [
        'han', 'hin', 'nan', 'ang', 'ing',
        'an', 'in', 'ng'
    ]
    
    # Reduplication pattern (e.g., "kaka", "sisi", "tata")
    REDUPLICATION_PATTERN = re.compile(r'^(\w{2,3})\1')
    
    # Valid Filipino word endings
    VALID_ENDINGS = ('a', 'e', 'i', 'o', 'u', 'ng', 'n', 'g', 'y', 'w')
    
    def get_morphological_score(self, word: str) -> float:
        """
        Score a word based on Filipino morphological patterns.
        
        Returns:
            Float between 0-1. Higher = more likely valid Filipino word.
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
        
        # Check for reduplication (common in Filipino)
        if self.REDUPLICATION_PATTERN.match(word):
            score += 0.1
        
        # Penalize very short words
        if len(word) <= 2:
            score -= 0.1
        
        # Bonus for valid Filipino word endings
        if word.endswith(self.VALID_ENDINGS):
            score += 0.05
        
        # Bonus for common Filipino patterns
        if self._has_common_pattern(word):
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _has_common_pattern(self, word: str) -> bool:
        """Check for common Filipino word patterns."""
        # CV-CV pattern is common
        if len(word) >= 4:
            vowels = set('aeiou')
            # Check alternating consonant-vowel
            cv_count = sum(
                1 for i in range(len(word)-1) 
                if (word[i] not in vowels and word[i+1] in vowels)
            )
            return cv_count >= 2
        return False
    
    def compare_candidates(self, word1: str, word2: str) -> Tuple[float, float]:
        """
        Compare morphological scores of two candidates.
        
        Returns:
            Tuple of normalized scores that sum to 1.
        """
        score1 = self.get_morphological_score(word1)
        score2 = self.get_morphological_score(word2)
        
        total = score1 + score2
        if total == 0:
            return 0.5, 0.5
        
        return score1 / total, score2 / total
    
    def get_root_word(self, word: str) -> str:
        """
        Attempt to extract root word by removing affixes.
        (Simplified - full implementation would need dictionary lookup)
        """
        word = word.lower()
        
        # Remove prefix
        for prefix in self.PREFIXES:
            if word.startswith(prefix):
                word = word[len(prefix):]
                break
        
        # Remove suffix
        for suffix in self.SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        return word
