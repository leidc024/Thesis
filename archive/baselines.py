"""
Baseline Models for Comparison
Implements simpler approaches to compare against our graph-based method.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Union
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm


class MaBaybayDefault:
    """
    Baseline 1: MaBaybay Default
    
    Simply selects the first candidate (no disambiguation).
    This represents the default behavior without any context awareness.
    
    Expected accuracy: ~38% on ambiguous words
    """
    
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
                result.append(item[0])  # Always first
            else:
                result.append(item)
        return result, {}
    
    def evaluate(self, test_data: List[Dict]) -> Tuple[Dict, List]:
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        for entry in tqdm(test_data, desc="MaBaybay Default"):
            gt = entry['ground_truth']
            candidates = entry['ocr_candidates']
            gt_words = gt.lower().split()
            
            predicted, _ = self.disambiguate(candidates)
            
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
            
            results.append({'gt': gt, 'pred': ' '.join(predicted)})
        
        return {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous else 0
        }, results


class LLMBaseline:
    """
    Baseline 3: LLM-based Disambiguation
    
    Uses an LLM (OpenAI, Google Gemini, or Ollama) to select the best candidate
    based on context. This is similar to bAI-bAI's LLM approach.
    
    Supported providers:
    - 'openai': OpenAI GPT-4/3.5 (requires OPENAI_API_KEY)
    - 'gemini': Google Gemini (requires GOOGLE_API_KEY)
    - 'ollama': Local Ollama (requires Ollama running locally)
    
    Expected accuracy: ~85-92% on ambiguous words (depending on model)
    """
    
    def __init__(self, provider: str = "gemini", model: str = None):
        """
        Initialize LLM baseline.
        
        Args:
            provider: 'openai', 'gemini', or 'ollama'
            model: Model name (defaults based on provider)
        """
        self.provider = provider.lower()
        self.client = None
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self.client = OpenAI(api_key=api_key)
                self.model = model or "gpt-3.5-turbo"
                print(f"[LLM Baseline] Using OpenAI {self.model}")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
                
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                genai.configure(api_key=api_key)
                # List available models and pick the best one
                self.model = model or "gemini-2.0-flash"  # Updated model name
                self.client = genai.GenerativeModel(self.model)
                print(f"[LLM Baseline] Using Google Gemini {self.model}")
            except ImportError:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
                
        elif self.provider == "ollama":
            try:
                import ollama
                self.client = ollama
                self.model = model or "llama3.2"
                # Test connection
                ollama.list()
                print(f"[LLM Baseline] Using Ollama {self.model}")
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            except Exception as e:
                raise ConnectionError(f"Could not connect to Ollama. Make sure it's running. Error: {e}")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'gemini', or 'ollama'")
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM and return response."""
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=0
                    )
                    return response.choices[0].message.content.strip()
                    
                elif self.provider == "gemini":
                    response = self.client.generate_content(prompt)
                    time.sleep(4.5)  # Rate limit: 15 requests/min = 1 per 4 seconds
                    return response.text.strip()
                    
                elif self.provider == "ollama":
                    response = self.client.generate(model=self.model, prompt=prompt)
                    return response['response'].strip()
                    
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    print(f"\n  [Rate limit hit, waiting {wait_time}s...]")
                    time.sleep(wait_time)
                    continue
                print(f"LLM query error: {e}")
                return ""
        
        return ""
    
    def disambiguate(
        self,
        ocr_candidates: List[Union[str, List[str]]],
        ground_truth: str = None
    ) -> Tuple[List[str], Dict]:
        """Disambiguate using LLM - ONE API call per sentence (batched)."""
        result = []
        debug = {}
        
        # Find all ambiguous positions
        ambiguous_positions = []
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list) and len(item) > 1:
                ambiguous_positions.append((pos, item))
        
        # If no ambiguities, just return first candidates
        if not ambiguous_positions:
            for item in ocr_candidates:
                result.append(item[0] if isinstance(item, list) else item)
            return result, debug
        
        # Build sentence with numbered placeholders
        context_words = []
        for i, item in enumerate(ocr_candidates):
            if isinstance(item, list) and len(item) > 1:
                # Find which ambiguous position this is
                amb_idx = next(j for j, (p, _) in enumerate(ambiguous_positions) if p == i)
                context_words.append(f"[{amb_idx + 1}]")
            else:
                context_words.append(item[0] if isinstance(item, list) else item)
        
        sentence_context = " ".join(context_words)
        
        # Build the candidates list for prompt
        candidates_list = []
        for idx, (pos, candidates) in enumerate(ambiguous_positions):
            cand_str = ", ".join(f'"{c}"' for c in candidates)
            candidates_list.append(f"  [{idx + 1}]: {cand_str}")
        
        candidates_prompt = "\n".join(candidates_list)
        
        # Single prompt for ALL ambiguities in this sentence
        prompt = f"""You are an expert linguist specializing in Filipino/Tagalog language and Baybayin script.

BACKGROUND:
Baybayin is the ancient Filipino writing system. When converting Baybayin to Latin script, certain characters are ambiguous:
- ᜇ (da/ra) can be either 'd' or 'r' 
- ᜁ (e/i) can be either 'e' or 'i'
- ᜂ (o/u) can be either 'o' or 'u'

TASK:
An OCR system read Baybayin text and produced multiple candidate words at positions marked [1], [2], etc.
Choose the CORRECT Filipino word for each position based on context.

Sentence: "{sentence_context}"

Candidates for each position:
{candidates_prompt}

Based on Filipino grammar, semantics, and common usage, which word is correct for each position?

IMPORTANT: Reply with ONLY the words separated by commas in order, nothing else.
Example format: word1, word2, word3"""
        
        response = self._query_llm(prompt)
        debug['prompt'] = prompt
        debug['response'] = response
        
        # Parse response - extract words
        response_words = [w.strip().strip('"\'.,') for w in response.split(',')]
        
        # Build final result
        amb_idx = 0
        for i, item in enumerate(ocr_candidates):
            if isinstance(item, list) and len(item) > 1:
                # Get LLM's choice for this position
                if amb_idx < len(response_words):
                    llm_choice = response_words[amb_idx].lower()
                    # Find matching candidate
                    selected = item[0]  # Default
                    for cand in item:
                        if cand.lower() == llm_choice or llm_choice in cand.lower() or cand.lower() in llm_choice:
                            selected = cand
                            break
                    result.append(selected)
                    debug[i] = {'candidates': item, 'llm_choice': llm_choice, 'selected': selected}
                else:
                    result.append(item[0])
                amb_idx += 1
            else:
                result.append(item[0] if isinstance(item, list) else item)
        
        return result, debug
    
    def evaluate(self, test_data: List[Dict], limit: int = None) -> Tuple[Dict, List]:
        """
        Evaluate LLM on test data.
        
        Args:
            test_data: List of test entries
            limit: Optional limit on number of entries (LLM calls can be slow/expensive)
        """
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        data_to_eval = test_data[:limit] if limit else test_data
        
        for entry in tqdm(data_to_eval, desc=f"LLM ({self.provider})"):
            gt = entry['ground_truth']
            candidates = entry['ocr_candidates']
            gt_words = gt.lower().split()
            
            predicted, debug = self.disambiguate(candidates)
            
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
        
        return {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous else 0
        }, results
class EmbeddingOnly:
    """
    Baseline 2: Embedding-Only (WE-Only)
    
    Uses only RoBERTa embeddings for semantic similarity.
    No frequency, co-occurrence, or morphology features.
    
    This is similar to bAI-bAI's WE-Only approach.
    Expected accuracy: ~66-77% on ambiguous words
    """
    
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
        # Build context
        context = ground_truth if ground_truth else ' '.join(
            c[0] if isinstance(c, list) else c for c in ocr_candidates
        )
        context_emb = self.get_embedding(context)
        
        result = []
        debug = {}
        
        for pos, item in enumerate(ocr_candidates):
            if isinstance(item, list):
                # Score by semantic similarity only
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
    
    def evaluate(self, test_data: List[Dict]) -> Tuple[Dict, List]:
        total_words = 0
        correct_words = 0
        total_ambiguous = 0
        correct_ambiguous = 0
        results = []
        
        for entry in tqdm(test_data, desc="Embedding-Only"):
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
            
            results.append({'gt': gt, 'pred': ' '.join(predicted)})
        
        return {
            'total_words': total_words,
            'correct_words': correct_words,
            'total_accuracy': correct_words / total_words if total_words else 0,
            'total_ambiguous': total_ambiguous,
            'correct_ambiguous': correct_ambiguous,
            'ambiguous_accuracy': correct_ambiguous / total_ambiguous if total_ambiguous else 0
        }, results
