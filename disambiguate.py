"""
Disambiguate MaBaybay OCR candidates.
Called from MATLAB: system('python disambiguate.py input.json')

Input JSON format:
[
    ["dito", "rito"],     // ambiguous word - multiple candidates
    ["ang"],              // unambiguous word - single candidate  
    ["lugar", "logar"],   // ambiguous word
    ...
]

Output: prints disambiguated sentence to stdout (MATLAB captures this)
"""

import sys
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def disambiguate(candidates_json: str) -> str:
    """Load candidates and disambiguate."""
    import io
    import warnings
    
    # Suppress all warnings and progress bars before importing anything
    warnings.filterwarnings('ignore')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm progress bars
    
    # Disable HuggingFace Hub warnings
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
    
    # Get the directory where this script lives (Thesis folder)
    script_dir = Path(__file__).parent.resolve()
    
    # Suppress BOTH stdout and stderr during initialization
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()  # Capture to nowhere
    sys.stderr = io.StringIO()  # Also capture stderr
    
    try:
        from src.disambiguator import BaybayinDisambiguator
        
        # Load candidates from JSON file
        with open(candidates_json, 'r', encoding='utf-8') as f:
            raw_candidates = json.load(f)
        
        # Convert format: single-item lists become strings
        candidates = []
        for item in raw_candidates:
            if isinstance(item, list) and len(item) == 1:
                candidates.append(item[0])  # Unambiguous
            else:
                candidates.append(item)     # Ambiguous (list) or already string
        
        # Count ambiguous positions
        num_ambiguous = sum(1 for c in candidates if isinstance(c, list))
        num_words = len(candidates)
        
        # For single-word input or all-ambiguous, use frequency-heavy weights
        # because semantic context is unreliable without surrounding words
        if num_words == 1 or num_ambiguous == num_words:
            # Single word: rely primarily on frequency (real words beat non-words)
            weights = {
                'semantic': 0.0,      # Reduce - no useful context
                'frequency': 0.1,     # Increase - prefer corpus words
                'cooccurrence': 0.0,  # No adjacent words
                'morphology': 0.0     # Morphological patterns help
            }
        else:
            # Normal multi-word sentence: use default balanced weights
            weights = None  # Use BaybayinDisambiguator defaults
        
        # Initialize model with absolute paths to corpus files
        model = BaybayinDisambiguator(
            corpus_files=[
                str(script_dir / "corpus" / "Tagalog_Literary_Text.txt"),
                str(script_dir / "corpus" / "Tagalog_Religious_Text.txt"),
                str(script_dir / "corpus" / "Tagalog_Balita_Texts_Balanced.txt")
            ],
            weights=weights
        )
        
        # Disambiguate
        result, _ = model.disambiguate(candidates)
        
    finally:
        # Always restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return ' '.join(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python disambiguate.py candidates.json", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        result = disambiguate(input_file)
        print(result)  # MATLAB captures stdout
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
