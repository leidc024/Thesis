"""
Step-by-step walkthrough of Mean-Pooled Cosine Similarity scoring.

Demonstrates how the OLD method scored "bote" vs "buti" for:
  "Ang malaking bote ng tubig ay nasa lamesa."

Companion to explain_mlm_pll.py — shows why cosine similarity was weaker.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  MEAN-POOLED COSINE SIMILARITY — STEP-BY-STEP DEMO")
print("  (The old semantic scoring method)")
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
position = 2

print("=" * 70)
print("  THE PROBLEM (Same as MLM PLL demo)")
print("=" * 70)
print(f"""
Sentence: "{sentence}"
Candidates: {candidates}

In the OLD method, we used cosine similarity of mean-pooled embeddings
to measure how semantically similar a candidate word is to the context.

Let's walk through exactly how that works.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: BUILD THE CONTEXT STRING
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 1: BUILD THE CONTEXT (without the ambiguous word)")
print("=" * 70)

words = sentence.replace(".", "").lower().split()
context_words = words[:position] + words[position+1:]
context = ' '.join(context_words)

print(f"""
  Original sentence:  {' '.join(words)}
  Ambiguous position: index {position} → '{words[position]}'
  
  We REMOVE the ambiguous word to build the context:
    Context words: {context_words}
    Context string: "{context}"
  
  Why? Because we don't want to bias toward any candidate.
  The context is the same for ALL candidates.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: GET CONTEXT EMBEDDING (mean pooling)
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 2: GET CONTEXT EMBEDDING (Mean Pooling)")
print("=" * 70)

print(f"""
  Mean pooling works like this:
  1. Feed the context into RoBERTa
  2. Get the hidden state (embedding) for EACH token
  3. Average all token embeddings into ONE vector
  
  This single vector is supposed to represent the "meaning" of the context.
""")

# Tokenize context
ctx_inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=128)
ctx_token_ids = ctx_inputs['input_ids'][0]
ctx_tokens = [tokenizer.decode([t]) for t in ctx_token_ids.tolist()]

print(f"  Context: \"{context}\"")
print(f"\n  Tokenized into {len(ctx_tokens)} tokens:")
for i, (tid, tok) in enumerate(zip(ctx_token_ids.tolist(), ctx_tokens)):
    print(f"    Position {i:2d}: {tok:15s}  (ID: {tid})")

with torch.no_grad():
    outputs = model(**ctx_inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1]  # Shape: [1, num_tokens, 768]
    
    mask = ctx_inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    sum_emb = torch.sum(embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    context_embedding = (sum_emb / sum_mask).cpu().numpy().flatten()

print(f"""
  Each token gets a 768-dimensional embedding vector from RoBERTa.
  
  RoBERTa output shape: [1, {embeddings.shape[1]} tokens, 768 dimensions]
  
  Mean pooling:
    • Sum all {embeddings.shape[1]} token embeddings element-wise
    • Divide by {embeddings.shape[1]} (number of tokens)
    • Result: 1 vector of 768 dimensions
  
  Context embedding (first 10 of 768 values):
    {context_embedding[:10].round(4)}...
  
  This ONE vector represents the entire context meaning.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: GET CANDIDATE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 3: GET CANDIDATE EMBEDDINGS (Same Process)")
print("=" * 70)

print("""
  Now we do the EXACT SAME mean-pooling process for each candidate word
  — but we're embedding the word ALONE, without any context.
""")

candidate_embeddings = {}
for cand in candidates:
    cand_inputs = tokenizer(cand, return_tensors="pt", truncation=True, max_length=128)
    cand_token_ids = cand_inputs['input_ids'][0]
    cand_tokens = [tokenizer.decode([t]) for t in cand_token_ids.tolist()]
    
    with torch.no_grad():
        cand_outputs = model(**cand_inputs, output_hidden_states=True)
        cand_hidden = cand_outputs.hidden_states[-1]
        cand_mask = cand_inputs['attention_mask'].unsqueeze(-1).expand(cand_hidden.size()).float()
        cand_sum = torch.sum(cand_hidden * cand_mask, dim=1)
        cand_sum_mask = torch.clamp(cand_mask.sum(dim=1), min=1e-9)
        cand_emb = (cand_sum / cand_sum_mask).cpu().numpy().flatten()
    
    candidate_embeddings[cand] = cand_emb
    
    print(f"  Candidate: \"{cand}\"")
    print(f"    Tokens: {cand_tokens}  ({len(cand_tokens)} tokens)")
    print(f"    Embedding (first 10 of 768): {cand_emb[:10].round(4)}...")
    print()

print("""  ⚠️  KEY PROBLEM: Each candidate is embedded IN ISOLATION.
     The word "bote" gets the same embedding regardless of whether 
     the sentence is about water bottles or wine bottles.
     It captures the word's GENERAL meaning, not its fit in THIS context.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 4: COMPUTE COSINE SIMILARITY")
print("=" * 70)

print("""
  Cosine similarity measures the angle between two vectors.
  
  Formula:  cos(A, B) = (A · B) / (||A|| × ||B||)
  
  Range: -1 to +1  (we clamp to 0 minimum)
    • 1.0  = identical direction (most similar)
    • 0.0  = perpendicular (unrelated)
    • -1.0 = opposite direction (most dissimilar)
""")

print("  Computing cosine similarity between each candidate and the context:\n")

cos_scores = {}
for cand in candidates:
    cand_emb = candidate_embeddings[cand]
    
    # Dot product
    dot_product = np.dot(context_embedding, cand_emb)
    # Norms
    norm_ctx = np.linalg.norm(context_embedding)
    norm_cand = np.linalg.norm(cand_emb)
    # Cosine similarity
    cos_sim = dot_product / (norm_ctx * norm_cand)
    cos_scores[cand] = max(0.0, float(cos_sim))
    
    print(f"  \"{cand}\":")
    print(f"    dot(context, {cand})  = {dot_product:.4f}")
    print(f"    ||context||          = {norm_ctx:.4f}")
    print(f"    ||\"{cand}\"||{' ' * (12-len(cand))} = {norm_cand:.4f}")
    print(f"    cosine similarity    = {dot_product:.4f} / ({norm_ctx:.4f} × {norm_cand:.4f})")
    print(f"                         = {cos_sim:.6f}")
    print(f"    clamped to [0, 1]    = {cos_scores[cand]:.6f}")
    print()

gap = abs(cos_scores['bote'] - cos_scores['buti'])
winner_cos = max(cos_scores, key=cos_scores.get)
loser_cos = min(cos_scores, key=cos_scores.get)

print(f"  ┌──────────────────────────────────────────┐")
print(f"  │  Cosine Similarity Results:               │")
print(f"  │    \"bote\":  {cos_scores['bote']:.6f}                     │")
print(f"  │    \"buti\":  {cos_scores['buti']:.6f}                     │")
print(f"  │    Gap:     {gap:.6f}                     │")
print(f"  │    Winner:  \"{winner_cos}\"                          │")
print(f"  └──────────────────────────────────────────┘")

print(f"""
  ⚠️  The gap is only {gap:.4f}!
  
  {"❌ WRONG! Cosine picks '" + winner_cos + "' but the correct answer is 'bote'!" if winner_cos != "bote" else "✓ Cosine picks the right answer, but barely."}
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: WHY THE SCORES ARE SO CLOSE
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 5: WHY ARE THE SCORES SO CLOSE?")
print("=" * 70)

print("""
  The fundamental issue with mean-pooled cosine similarity:
  
  1. CONTEXT EMBEDDING is an average of ALL tokens:
     "ang", "malaking", "ng", "tubig", "ay", "nasa", "lamesa"
     This averages words about water, tables, sizes — everything blurs.
  
  2. CANDIDATE EMBEDDING has no sentence context:
     "bote" is embedded as just the word "bote" — its general meaning.
     "buti" is embedded as just the word "buti" — its general meaning.
  
  3. Both are COMMON Filipino words, so both have embeddings that are
     reasonably close to any Filipino text context.
  
  The result? Both words score SIMILARLY because mean pooling creates
  such generic vectors that they can't distinguish fine-grained context.
  
  Think of it this way:
    • Cosine asks: "Is 'bote' related to Filipino text about big things
      with water on a table?" → Sure, somewhat.
    • Cosine asks: "Is 'buti' related to Filipino text about big things
      with water on a table?" → Sure, somewhat too.
    
  The general-purpose embeddings just aren't specific enough.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: SIDE-BY-SIDE COMPARISON WITH PLL
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 6: SIDE-BY-SIDE COMPARISON WITH MLM PLL")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │           COSINE SIMILARITY vs MLM PLL                         │
  ├─────────────────┬─────────────────────┬────────────────────────┤
  │                 │ Cosine Similarity   │  MLM PLL               │
  ├─────────────────┼─────────────────────┼────────────────────────┤
  │ What it does    │ Compare embedding   │  Mask word, ask model  │
  │                 │ vectors via angle   │  "what goes here?"     │
  ├─────────────────┼─────────────────────┼────────────────────────┤
  │ Context used    │ Context WITHOUT     │  Full sentence WITH    │
  │                 │ candidate word      │  candidate inserted    │
  ├─────────────────┼─────────────────────┼────────────────────────┤
  │ Candidate       │ Embedded ALONE,     │  Evaluated IN context  │
  │ representation  │ no context          │  of the sentence       │
  ├─────────────────┼─────────────────────┼────────────────────────┤
  │ "bote" score    │   {cos_scores['bote']:.6f}          │  ~1.0000 (rank #6)     │
  │ "buti" score    │   {cos_scores['buti']:.6f}          │  ~0.0000 (rank #7607)  │
  ├─────────────────┼─────────────────────┼────────────────────────┤
  │ Gap             │   {gap:.6f}          │  ~1.0000               │
  │                 │   (tiny!)           │  (massive!)            │
  ├─────────────────┼─────────────────────┼────────────────────────┤
  │ Signal strength │   WEAK              │  STRONG                │
  │                 │   Can be dominated  │  Dominates over        │
  │                 │   by frequency bias │  frequency bias        │
  └─────────────────┴─────────────────────┴────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 7: VISUALIZE THE EMBEDDING SPACE
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 7: VISUALIZE THE PROBLEM")
print("=" * 70)

# Compute angle between vectors
cos_bote = cos_scores['bote']
cos_buti = cos_scores['buti']
angle_bote = np.degrees(np.arccos(np.clip(cos_bote, -1, 1)))
angle_buti = np.degrees(np.arccos(np.clip(cos_buti, -1, 1)))

print(f"""
  In 768-dimensional space, all three vectors point in roughly
  similar directions (they're all Filipino text after all):
  
  Angle between context and "bote": {angle_bote:.2f}°
  Angle between context and "buti": {angle_buti:.2f}°
  Difference in angle:              {abs(angle_bote - angle_buti):.2f}°
  
  That's only ~{abs(angle_bote - angle_buti):.1f}° difference in a 768-dimensional space!
  
  Imagine trying to tell two stars apart that are {abs(angle_bote - angle_buti):.1f}° away 
  in the sky — they look almost identical.
  
  
  COSINE APPROACH (vague, both words look similar):
  
       context ──────────────→  (average of all tokens)
       "bote"  ──────────────→  (general "bottle" meaning)  
       "buti"  ──────────────→  (general "good" meaning)
                                ↑ All vectors roughly same direction
  
  
  PLL APPROACH (precise, directly tests fit):
  
       "ang malaking [MASK] ng tubig ay nasa lamesa"
                       │
                       ▼
              Model predicts:
              #1 bahagi,  #2 tasa,  #3 basket...
              #6 bote  ← Good fit!  (prob: 3.2%)
                  ...
              #7607 buti ← Terrible fit! (prob: 0.00006%)
""")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"""
  Sentence: "Ang malaking ___ ng tubig ay nasa lamesa."
  Correct answer: "bote" (bottle)
  
  COSINE SIMILARITY (old method):
    • Embeds context and candidates separately
    • Compares them by vector angle  
    • "bote" = {cos_scores['bote']:.6f},  "buti" = {cos_scores['buti']:.6f}
    • Gap: {gap:.6f} — too small to be reliable
    • {"❌ Actually picks WRONG answer!" if winner_cos != "bote" else "✓ Gets it right, but barely — easily overridden by frequency"}
  
  MLM PLL (new method):
    • Puts each candidate INTO the sentence
    • Asks model to predict the masked word
    • "bote" = ~1.0000, "buti" = ~0.0000
    • Gap: ~1.0000 — overwhelming signal
    • ✓ Gets it right with high confidence
  
  The key insight: PLL asks the RIGHT question.
    Cosine: "Are these words vaguely related?" → Both somewhat related.
    PLL:    "Would a native speaker use THIS word HERE?" → Clearly "bote".
""")
