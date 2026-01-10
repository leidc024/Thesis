"""
Baseline Competitor Models for Baybayin Disambiguation
Implements two simple baselines to compare against our graph-based approach:

1. MaBaybay-OCR Default: Simply picks the first candidate (what OCR does by default)
2. Embedding-Only (WE-Only): Uses word embeddings similarity only (replicates bAI-bAI WE method)
"""

import json
import sys
import io
from pathlib import Path
from typing import List, Dict, Tuple, Union

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configuration
MODEL_NAME = "jcblaise/roberta-tagalog-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("dataset")
CANDIDATES_FILE = DATA_DIR / "processed" / "candidates_results_v1.json"
SPLITS_DIR = DATA_DIR / "splits"


class MaBaybayDefault:
    """
    Baseline 1: MaBaybay-OCR Default
    Simply selects the first candidate from the OCR output.
    This is what happens when no disambiguation is applied.
    """
    
    def __init__(self):
        print("[MaBaybay Default] Initialized - picks first candidate always")
    
    def disambiguate_sentence(
        self, 
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        """Select first candidate at each position."""
        disambiguated = []
        debug_info = {'selected': {}}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                # Ambiguous - just take first candidate
                selected = item[0]
                disambiguated.append(selected)
                debug_info['selected'][pos] = selected
            else:
                disambiguated.append(item)
        
        return disambiguated, debug_info
    
    def evaluate(self, test_data: List[Dict], verbose: bool = False) -> Tuple[Dict, List]:
        """Evaluate on test data."""
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        for entry in tqdm(test_data, desc="MaBaybay Default"):
            ground_truth = entry['ground_truth']
            ocr_candidates = entry['ocr_candidates']
            gt_words = ground_truth.lower().split()
            
            predicted, debug_info = self.disambiguate_sentence(ocr_candidates, ground_truth)
            
            entry_result = {'ground_truth': ground_truth, 'predicted': ' '.join(predicted), 'details': []}
            
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
            
            results.append(entry_result)
        
        metrics = {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words > 0 else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous > 0 else 0
        }
        
        return metrics, results


class EmbeddingOnlyDisambiguator:
    """
    Baseline 2: Embedding-Only (WE-Only)
    Replicates the Word Embedding only approach from bAI-bAI.
    Uses RoBERTa embeddings and selects the candidate most similar to context.
    No frequency, co-occurrence, or morphology features.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[Embedding-Only] Loading model: {model_name}")
        print(f"[Embedding-Only] Device: {DEVICE}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        print("[Embedding-Only] Model loaded - uses semantic similarity only")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get mean-pooled RoBERTa embedding for text."""
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
    
    def disambiguate_sentence(
        self,
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        """Disambiguate using embedding similarity only."""
        # Build context from first candidates or ground truth
        if ground_truth:
            context = ground_truth
        else:
            context = ' '.join(
                item[0] if isinstance(item, list) else item
                for item in ocr_candidates
            )
        
        context_embedding = self.get_embedding(context)
        
        disambiguated = []
        debug_info = {'selected': {}, 'scores': {}}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                # Score each candidate by similarity to context
                best_candidate = None
                best_score = -1
                scores = {}
                
                for candidate in item:
                    candidate_embedding = self.get_embedding(candidate)
                    similarity = cosine_similarity(
                        candidate_embedding.reshape(1, -1),
                        context_embedding.reshape(1, -1)
                    )[0, 0]
                    scores[candidate] = float(similarity)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_candidate = candidate
                
                disambiguated.append(best_candidate)
                debug_info['selected'][pos] = best_candidate
                debug_info['scores'][pos] = scores
            else:
                disambiguated.append(item)
        
        return disambiguated, debug_info
    
    def evaluate(self, test_data: List[Dict], verbose: bool = False) -> Tuple[Dict, List]:
        """Evaluate on test data."""
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        for entry in tqdm(test_data, desc="Embedding-Only"):
            ground_truth = entry['ground_truth']
            ocr_candidates = entry['ocr_candidates']
            gt_words = ground_truth.lower().split()
            
            predicted, debug_info = self.disambiguate_sentence(ocr_candidates, ground_truth)
            
            entry_result = {'ground_truth': ground_truth, 'predicted': ' '.join(predicted), 'details': []}
            
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
            
            results.append(entry_result)
        
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


def load_splits():
    """Load train/validation/test splits."""
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        filepath = SPLITS_DIR / f"{split_name}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            splits[split_name] = json.load(f)
    return splits


def create_candidates_for_split(split_data, all_candidates):
    """Match split sentences with their OCR candidates."""
    candidates_lookup = {
        entry['ground_truth'].strip().lower(): entry
        for entry in all_candidates
    }
    
    matched = []
    for item in split_data:
        if isinstance(item, dict):
            sentence = item.get('sentence', item.get('text', '')).strip()
        else:
            sentence = str(item).strip()
        
        key = sentence.lower()
        if key in candidates_lookup:
            matched.append(candidates_lookup[key].copy())
    
    return matched


def main():
    """Run comparison of all baseline models."""
    print("=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    all_candidates = load_dataset(str(CANDIDATES_FILE))
    splits = load_splits()
    test_data = create_candidates_for_split(splits['test'], all_candidates)
    print(f"Test set: {len(test_data)} sentences")
    print()
    
    results_summary = {}
    
    # ==========================================
    # Baseline 1: MaBaybay-OCR Default
    # ==========================================
    print("=" * 70)
    print("BASELINE 1: MaBaybay-OCR Default (First Candidate)")
    print("=" * 70)
    
    model1 = MaBaybayDefault()
    metrics1, _ = model1.evaluate(test_data)
    
    print(f"\nResults:")
    print(f"  Total Word Accuracy:     {metrics1['total_accuracy']:.2%}")
    print(f"  Ambiguous Word Accuracy: {metrics1['ambiguous_accuracy']:.2%}")
    print(f"  Correct Ambiguous:       {metrics1['correct_ambiguous']}/{metrics1['total_ambiguous']}")
    
    results_summary['MaBaybay Default'] = metrics1['ambiguous_accuracy']
    print()
    
    # ==========================================
    # Baseline 2: Embedding-Only (WE-Only)
    # ==========================================
    print("=" * 70)
    print("BASELINE 2: Embedding-Only (WE-Only Method)")
    print("=" * 70)
    
    model2 = EmbeddingOnlyDisambiguator()
    metrics2, _ = model2.evaluate(test_data)
    
    print(f"\nResults:")
    print(f"  Total Word Accuracy:     {metrics2['total_accuracy']:.2%}")
    print(f"  Ambiguous Word Accuracy: {metrics2['ambiguous_accuracy']:.2%}")
    print(f"  Correct Ambiguous:       {metrics2['correct_ambiguous']}/{metrics2['total_ambiguous']}")
    
    results_summary['Embedding-Only (WE)'] = metrics2['ambiguous_accuracy']
    print()
    
    # ==========================================
    # Summary Comparison
    # ==========================================
    print("=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print()
    print("Method                              | Ambiguous Accuracy | Notes")
    print("-" * 75)
    print(f"MaBaybay Default (first candidate)  |      {results_summary['MaBaybay Default']:.2%}         | No disambiguation")
    print(f"Embedding-Only (WE-Only)            |      {results_summary['Embedding-Only (WE)']:.2%}         | Semantic similarity only")
    print(f"bAI-bAI WE-Only (reported)          |      77.46%         | From paper")
    print(f"bAI-bAI LLM (reported)              |      90.52%         | From paper")
    print(f"Our Graph+Features v2               |      87.23%         | Sem+Freq+Cooc+Morph")
    print()
    
    # Save results
    results_file = DATA_DIR / "results" / "baseline_comparison.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mababay_default': {
                'ambiguous_accuracy': float(metrics1['ambiguous_accuracy']),
                'total_accuracy': float(metrics1['total_accuracy']),
                'correct': metrics1['correct_ambiguous'],
                'total': metrics1['total_ambiguous']
            },
            'embedding_only': {
                'ambiguous_accuracy': float(metrics2['ambiguous_accuracy']),
                'total_accuracy': float(metrics2['total_accuracy']),
                'correct': metrics2['correct_ambiguous'],
                'total': metrics2['total_ambiguous']
            }
        }, f, indent=2)
    print(f"[OK] Results saved to {results_file}")


if __name__ == "__main__":
    main()
