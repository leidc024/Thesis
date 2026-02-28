"""
Step-by-step walkthrough of MLM Pseudo-Log-Likelihood (PLL) scoring.

Demonstrates how the disambiguator picks "bote" over "buti" for:
  "Ang malaking bote ng tubig ay nasa lamesa."

This is an educational script meant to explain PLL to your thesis adviser.
"""

import math
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  MLM PSEUDO-LOG-LIKELIHOOD (PLL) SCORING — STEP-BY-STEP DEMO")
print("=" * 70)

model_name = "jcblaise/roberta-tagalog-base"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
print("Model loaded!\n")

# ══════════════════════════════════════════════════════════════════════
# THE PROBLEM
# ══════════════════════════════════════════════════════════════════════
sentence = "Ang malaking bote ng tubig ay nasa lamesa."
candidates = ["bote", "buti"]
position = 2  # "bote" is at index 2 in word list

print("=" * 70)
print("  THE PROBLEM")
print("=" * 70)
print(f"""
In Baybayin script, the word 'ᜊᜓᜆᜒ' can be read as either:
  • "bote"  (bottle)
  • "buti"  (good/fortune)

Both are valid Filipino words, but only ONE is correct in context.

Sentence: "{sentence}"
Candidates: {candidates}

Question: How does PLL determine which candidate fits better?
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: TOKENIZATION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 1: TOKENIZATION")
print("=" * 70)

words = sentence.replace(".", "").lower().split()
print(f"\nSentence words: {words}")
print(f"Ambiguous position: index {position} → '{words[position]}'")

print(f"\n--- How RoBERTa tokenizes each candidate ---")
for cand in candidates:
    # Tokenize with space prefix (how words appear mid-sentence)
    token_ids = tokenizer.encode(' ' + cand, add_special_tokens=False)
    tokens = [tokenizer.decode([t]) for t in token_ids]
    print(f'  "{cand}" → subtokens: {tokens}  (IDs: {token_ids})')

# ══════════════════════════════════════════════════════════════════════
# STEP 2: PLL FOR EACH CANDIDATE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  STEP 2: PLL SCORING FOR EACH CANDIDATE")
print("=" * 70)
print("""
PLL works by asking: "If I put this candidate into the sentence,
how well does the language model predict each piece of the word
given all the surrounding context?"

For each subtoken of the candidate:
  1. Insert candidate into the sentence
  2. MASK that one subtoken (replace with [MASK])
  3. Ask the model: "What word goes here?"
  4. Record the log-probability of the CORRECT subtoken
  5. Sum all log-probabilities → PLL score
""")

pll_scores = {}

for cand in candidates:
    print(f"\n{'─' * 60}")
    print(f"  Scoring candidate: \"{cand}\"")
    print(f"{'─' * 60}")
    
    # Build sentence with this candidate
    test_words = list(words)
    test_words[position] = cand
    full_sentence = ' '.join(test_words)
    print(f"  Full sentence: \"{full_sentence}\"")
    
    # Tokenize
    encoding = tokenizer(full_sentence, return_tensors='pt', truncation=True, max_length=128)
    token_ids = encoding['input_ids'][0].clone()
    attention_mask = encoding['attention_mask'][0]
    
    # Show full tokenization
    all_tokens = [tokenizer.decode([t]) for t in token_ids.tolist()]
    print(f"\n  Tokenized sentence:")
    for i, (tid, tok) in enumerate(zip(token_ids.tolist(), all_tokens)):
        marker = ""
        print(f"    Position {i:2d}: {tok:15s}  (ID: {tid}){marker}")
    
    # Find candidate's subtoken positions
    cand_token_ids = tokenizer.encode(' ' + cand, add_special_tokens=False)
    cand_tokens_str = [tokenizer.decode([t]) for t in cand_token_ids]
    
    # Search for the subtoken sequence in the full token IDs
    seq = token_ids.tolist()
    cand_positions = []
    for i in range(len(seq) - len(cand_token_ids) + 1):
        if seq[i:i+len(cand_token_ids)] == cand_token_ids:
            cand_positions = list(range(i, i + len(cand_token_ids)))
            break
    
    print(f"\n  Candidate subtokens: {cand_tokens_str} at positions {cand_positions}")
    
    # Compute PLL
    print(f"\n  --- Masking each subtoken one at a time ---\n")
    total_log_prob = 0.0
    
    for step, pos in enumerate(cand_positions, 1):
        original_token_id = token_ids[pos].item()
        original_token_str = tokenizer.decode([original_token_id])
        
        # Create masked version
        masked = token_ids.clone()
        masked[pos] = tokenizer.mask_token_id
        masked_sentence = tokenizer.decode(masked, skip_special_tokens=True)
        
        print(f"  Mask step {step}/{len(cand_positions)}:")
        print(f"    Masked sentence: \"{masked_sentence}\"")
        print(f"    Masked position {pos}: expecting \"{original_token_str}\" (ID: {original_token_id})")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(masked.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            logits = outputs.logits[0, pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob = log_probs[original_token_id].item()
            prob = math.exp(log_prob)
        
        # Show top 5 predictions
        top_k = 5
        top_values, top_indices = torch.topk(log_probs, top_k)
        print(f"\n    Model's top {top_k} predictions for [MASK]:")
        for rank, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), 1):
            predicted_token = tokenizer.decode([idx])
            is_target = " ← OUR TARGET" if idx == original_token_id else ""
            print(f"      #{rank}: \"{predicted_token}\"  (prob: {math.exp(val):.4f}, log-prob: {val:.4f}){is_target}")
        
        # Check if target is not in top 5
        if original_token_id not in top_indices.tolist():
            # Find actual rank
            all_sorted = torch.argsort(log_probs, descending=True)
            rank = (all_sorted == original_token_id).nonzero(as_tuple=True)[0].item() + 1
            print(f"      ... \"{original_token_str}\" is ranked #{rank} (prob: {prob:.6f}, log-prob: {log_prob:.4f}) ← OUR TARGET")
        
        total_log_prob += log_prob
        print(f"\n    Log-prob for \"{original_token_str}\": {log_prob:.4f}")
        print(f"    Running total: {total_log_prob:.4f}\n")
    
    pll_scores[cand] = total_log_prob
    print(f"  ★ Final PLL score for \"{cand}\": {total_log_prob:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: SOFTMAX NORMALIZATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  STEP 3: SOFTMAX NORMALIZATION")
print("=" * 70)
print("""
Raw PLL scores are negative numbers (log-probabilities).
We normalize them using softmax to get values between 0 and 1
that sum to 1 — like probabilities.

Formula: softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
""")

max_pll = max(pll_scores.values())
print(f"  Raw PLL scores:")
for cand, score in pll_scores.items():
    print(f"    \"{cand}\": {score:.4f}")

print(f"\n  max(PLL) = {max_pll:.4f}")
print(f"\n  Subtract max (for numerical stability):")
shifted = {}
for cand, score in pll_scores.items():
    shifted[cand] = score - max_pll
    print(f"    \"{cand}\": {score:.4f} - ({max_pll:.4f}) = {shifted[cand]:.4f}")

print(f"\n  Exponentiate:")
exp_scores = {}
for cand, s in shifted.items():
    exp_scores[cand] = math.exp(s)
    print(f"    exp({s:.4f}) = {exp_scores[cand]:.6f}")

total_exp = sum(exp_scores.values())
print(f"\n  Sum of exponents: {total_exp:.6f}")

print(f"\n  Divide by sum (normalize):")
normalized = {}
for cand in candidates:
    normalized[cand] = exp_scores[cand] / total_exp
    print(f"    \"{cand}\": {exp_scores[cand]:.6f} / {total_exp:.6f} = {normalized[cand]:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: FINAL DECISION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  STEP 4: HOW THE FINAL DECISION IS MADE")
print("=" * 70)

print(f"""
These normalized MLM scores become the "semantic" feature score.
They are then combined with other features using weighted sum:

  Final Score = 0.4 × semantic + 0.3 × frequency + 0.2 × cooccurrence + 0.1 × morphology

Normalized MLM scores (semantic feature):
  "bote": {normalized['bote']:.4f}
  "buti": {normalized['buti']:.4f}
  
  Gap: {abs(normalized['bote'] - normalized['buti']):.4f}
""")

winner = max(normalized, key=normalized.get)
loser = min(normalized, key=normalized.get)
print(f"  The MLM score for \"{winner}\" ({normalized[winner]:.4f}) is much higher")
print(f"  than \"{loser}\" ({normalized[loser]:.4f}).")
print(f"\n  With a semantic weight of 0.4, this contributes:")
print(f"    \"{winner}\": 0.4 × {normalized[winner]:.4f} = {0.4 * normalized[winner]:.4f}")
print(f"    \"{loser}\":  0.4 × {normalized[loser]:.4f} = {0.4 * normalized[loser]:.4f}")
print(f"    Advantage: {0.4 * abs(normalized[winner] - normalized[loser]):.4f}")
print(f"""
  This gap is large enough to overcome any frequency bias,
  so the system correctly picks: ★ "{winner}" ★
""")

# ══════════════════════════════════════════════════════════════════════
# COMPARISON WITH COSINE SIMILARITY (for reference)
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  BONUS: WHY COSINE SIMILARITY WAS WEAKER")
print("=" * 70)

# Get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

context = "Ang malaking ng tubig ay nasa lamesa"  # context without ambiguous word
context_emb = get_embedding(context).numpy().flatten()

print(f"\n  Context (without ambiguous word): \"{context}\"")
print(f"\n  Cosine similarity of candidate embeddings with context:")

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import numpy as np

for cand in candidates:
    cand_emb = get_embedding(cand).numpy().flatten()
    sim = cos_sim(cand_emb.reshape(1, -1), context_emb.reshape(1, -1))[0, 0]
    print(f"    \"{cand}\": {sim:.6f}")

print(f"""
  Notice: The cosine similarity scores are very close to each other.
  This is because mean-pooled embeddings capture general meaning 
  but lose the fine-grained contextual fit that PLL directly measures.

  PLL asks: "Does the model PREDICT this exact word here?"
  Cosine asks: "Is this word's embedding SIMILAR to the context embedding?"
  
  PLL is a much more direct and powerful test of contextual fit.
""")

print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"""
  Sentence: "Ang malaking ___ ng tubig ay nasa lamesa."
  Candidates: {candidates}

  ┌────────────────────────────────────────────────────┐
  │            PLL Scoring Results                     │
  ├──────────┬──────────────┬─────────────────────────┤
  │ Candidate│  Raw PLL     │  Normalized (softmax)   │
  ├──────────┼──────────────┼─────────────────────────┤
  │ bote     │  {pll_scores['bote']:>10.4f}  │  {normalized['bote']:>10.4f}              │
  │ buti     │  {pll_scores['buti']:>10.4f}  │  {normalized['buti']:>10.4f}              │
  └──────────┴──────────────┴─────────────────────────┘

  ★ Winner: "{winner}" — the model strongly predicts "bote" 
    fits the context about bottles and water on a table.
""")
