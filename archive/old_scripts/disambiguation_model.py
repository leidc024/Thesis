"""
Graph-Based Disambiguation Model for Baybayin OCR
Uses RoBERTa embeddings + Personalized PageRank for context-aware disambiguation.

Architecture:
1. Get RoBERTa embeddings for each candidate word
2. Build a graph where nodes are candidate words
3. Edges weighted by semantic similarity between candidates
4. Use Personalized PageRank to rank candidates based on context
5. Select highest-ranked candidate for each ambiguous position
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import List, Dict, Tuple, Union
from tqdm import tqdm

# Configuration
MODEL_NAME = "jcblaise/roberta-tagalog-base"  # Filipino RoBERTa model
CANDIDATES_FILE = "dataset/processed/candidates_results_v1.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaybayinDisambiguator:
    """
    Graph-based disambiguator using RoBERTa embeddings and Personalized PageRank.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the model and tokenizer."""
        print(f"Loading model: {model_name}")
        print(f"Device: {DEVICE}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def get_word_embedding(self, word: str, context: str = None) -> np.ndarray:
        """
        Get RoBERTa embedding for a word, optionally within a context.
        
        Args:
            word: The word to embed
            context: Optional sentence context for contextual embedding
            
        Returns:
            numpy array of shape (768,) - the word embedding
        """
        if context:
            # Get contextual embedding by encoding the full sentence
            # and extracting the embedding for the target word
            text = context
        else:
            text = word
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(DEVICE)
            
            outputs = self.model(**inputs)
            
            # Use [CLS] token embedding as sentence/word representation
            # or mean pooling of all tokens
            embeddings = outputs.last_hidden_state
            
            # Mean pooling (excluding padding)
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding.cpu().numpy().flatten()
    
    def get_candidate_embeddings(self, candidates: List[str], context: str) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all candidate words within context.
        
        Args:
            candidates: List of candidate words
            context: The sentence context
            
        Returns:
            Dictionary mapping candidate words to their embeddings
        """
        embeddings = {}
        for candidate in candidates:
            # Create context with candidate word
            embeddings[candidate] = self.get_word_embedding(candidate, context)
        return embeddings
    
    def build_disambiguation_graph(
        self, 
        ocr_candidates: List[Union[str, List[str]]],
        context_embedding: np.ndarray
    ) -> nx.Graph:
        """
        Build a graph for disambiguation.
        
        Nodes: All candidate words from all positions
        Edges: Weighted by semantic similarity
        
        Args:
            ocr_candidates: List where each element is either a string (unambiguous)
                           or list of strings (ambiguous candidates)
            context_embedding: Embedding of the full sentence context
            
        Returns:
            NetworkX graph for PageRank computation
        """
        G = nx.Graph()
        
        # Collect all candidates with their positions
        all_candidates = []
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                for candidate in item:
                    all_candidates.append((pos, candidate))
                    G.add_node(f"{pos}_{candidate}", 
                              position=pos, 
                              word=candidate,
                              is_ambiguous=True)
            else:
                all_candidates.append((pos, item))
                G.add_node(f"{pos}_{item}", 
                          position=pos, 
                          word=item,
                          is_ambiguous=False)
        
        # Get embeddings for all candidates
        candidate_embeddings = {}
        for pos, candidate in all_candidates:
            node_id = f"{pos}_{candidate}"
            candidate_embeddings[node_id] = self.get_word_embedding(candidate)
        
        # Add edges between nodes at different positions
        # Weight by semantic similarity
        for i, (pos1, cand1) in enumerate(all_candidates):
            node1 = f"{pos1}_{cand1}"
            emb1 = candidate_embeddings[node1]
            
            for j, (pos2, cand2) in enumerate(all_candidates):
                if i >= j:  # Avoid duplicate edges
                    continue
                if pos1 == pos2:  # Don't connect candidates at same position
                    continue
                    
                node2 = f"{pos2}_{cand2}"
                emb2 = candidate_embeddings[node2]
                
                # Compute similarity
                similarity = cosine_similarity(
                    emb1.reshape(1, -1), 
                    emb2.reshape(1, -1)
                )[0, 0]
                
                # Only add edge if similarity is positive
                if similarity > 0:
                    G.add_edge(node1, node2, weight=similarity)
        
        return G, candidate_embeddings
    
    def disambiguate_sentence(
        self, 
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        """
        Disambiguate a sentence using graph-based approach.
        
        Args:
            ocr_candidates: OCR output with candidate lists for ambiguous words
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            Tuple of (disambiguated_words, debug_info)
        """
        # Get context embedding from ground truth or reconstructed sentence
        if ground_truth:
            context = ground_truth
        else:
            # Reconstruct using first candidate at each position
            context = ' '.join(
                item[0] if isinstance(item, list) else item 
                for item in ocr_candidates
            )
        
        context_embedding = self.get_word_embedding(context)
        
        # Build disambiguation graph
        G, candidate_embeddings = self.build_disambiguation_graph(
            ocr_candidates, context_embedding
        )
        
        # Apply Personalized PageRank
        # Personalization: bias towards words similar to context
        personalization = {}
        for node in G.nodes():
            emb = candidate_embeddings[node]
            similarity = cosine_similarity(
                emb.reshape(1, -1),
                context_embedding.reshape(1, -1)
            )[0, 0]
            personalization[node] = max(similarity, 0.01)  # Avoid zero
        
        # Normalize personalization
        total = sum(personalization.values())
        personalization = {k: v/total for k, v in personalization.items()}
        
        # Run PageRank
        try:
            pagerank_scores = nx.pagerank(
                G, 
                alpha=0.85,  # Damping factor
                personalization=personalization,
                weight='weight'
            )
        except:
            # Fallback if PageRank fails (e.g., disconnected graph)
            pagerank_scores = personalization
        
        # Select best candidate at each position
        disambiguated = []
        debug_info = {'scores': {}, 'selected': {}}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                # Ambiguous - select highest scoring candidate
                candidates_at_pos = [(f"{pos}_{c}", c) for c in item]
                scores = [(node_id, word, pagerank_scores.get(node_id, 0)) 
                         for node_id, word in candidates_at_pos]
                scores.sort(key=lambda x: -x[2])
                
                best_word = scores[0][1]
                disambiguated.append(best_word)
                
                debug_info['scores'][pos] = scores
                debug_info['selected'][pos] = best_word
            else:
                # Unambiguous - keep as is
                disambiguated.append(item)
        
        return disambiguated, debug_info
    
    def evaluate(self, test_data: List[Dict], verbose: bool = False) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of {ground_truth, ocr_candidates} dicts
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with evaluation metrics
        """
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
            entry_correct = 0
            entry_total = 0
            entry_ambiguous_correct = 0
            entry_ambiguous_total = 0
            
            for i, (pred, gt) in enumerate(zip(predicted, gt_words)):
                if i >= len(ocr_candidates):
                    break
                    
                is_ambiguous = isinstance(ocr_candidates[i], list)
                is_correct = pred.lower() == gt.lower()
                
                total_words += 1
                entry_total += 1
                if is_correct:
                    correct_words += 1
                    entry_correct += 1
                
                if is_ambiguous:
                    total_ambiguous += 1
                    entry_ambiguous_total += 1
                    if is_correct:
                        correct_ambiguous += 1
                        entry_ambiguous_correct += 1
            
            results.append({
                'ground_truth': ground_truth,
                'predicted': ' '.join(predicted),
                'correct': entry_correct,
                'total': entry_total,
                'ambiguous_correct': entry_ambiguous_correct,
                'ambiguous_total': entry_ambiguous_total
            })
            
            if verbose and entry_ambiguous_total > 0:
                print(f"\nGT: {ground_truth}")
                print(f"PR: {' '.join(predicted)}")
                print(f"Accuracy: {entry_correct}/{entry_total}, "
                      f"Ambiguous: {entry_ambiguous_correct}/{entry_ambiguous_total}")
        
        # Compute metrics
        metrics = {
            'total_accuracy': correct_words / total_words if total_words > 0 else 0,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous > 0 else 0,
            'total_words': total_words,
            'correct_words': correct_words,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
        }
        
        return metrics, results


def load_dataset(filepath: str) -> List[Dict]:
    """Load the candidates JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function to run disambiguation."""
    print("=" * 60)
    print("BAYBAYIN OCR DISAMBIGUATION MODEL")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading dataset...")
    data = load_dataset(CANDIDATES_FILE)
    print(f"✓ Loaded {len(data)} sentences")
    
    # Initialize model
    print()
    disambiguator = BaybayinDisambiguator()
    
    # Test on a few examples first
    print()
    print("=" * 60)
    print("TESTING ON SAMPLE SENTENCES")
    print("=" * 60)
    
    for entry in data[:5]:
        print(f"\nGround Truth: {entry['ground_truth']}")
        print(f"OCR Candidates: {entry['ocr_candidates']}")
        
        predicted, debug_info = disambiguator.disambiguate_sentence(
            entry['ocr_candidates'],
            entry['ground_truth']
        )
        
        print(f"Predicted: {' '.join(predicted)}")
        
        # Show disambiguation decisions
        for pos, selected in debug_info['selected'].items():
            scores = debug_info['scores'][pos]
            print(f"  Position {pos}: {[f'{w}:{s:.3f}' for _, w, s in scores]}")
    
    # Evaluate on full dataset
    print()
    print("=" * 60)
    print("FULL EVALUATION")
    print("=" * 60)
    
    metrics, results = disambiguator.evaluate(data[:100], verbose=False)  # Test on first 100
    
    print()
    print("RESULTS:")
    print(f"  Total Word Accuracy: {metrics['total_accuracy']:.2%}")
    print(f"  Ambiguous Word Accuracy: {metrics['ambiguous_accuracy']:.2%}")
    print(f"  Total Words: {metrics['total_words']}")
    print(f"  Correct Words: {metrics['correct_words']}")
    print(f"  Total Ambiguous: {metrics['total_ambiguous']}")
    print(f"  Correct Ambiguous: {metrics['correct_ambiguous']}")


if __name__ == "__main__":
    main()
