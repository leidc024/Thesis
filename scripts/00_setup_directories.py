"""
Setup Script: Creates the directory structure for dataset creation
Run this first before executing any other scripts.
"""

from pathlib import Path

# Directory structure
DIRECTORIES = [
    "dataset",
    "dataset/raw",
    "dataset/processed",
    "dataset/splits",
    "dataset/analysis",
    "dataset/documentation",
    "scripts",
]

def create_directories():
    """Create all necessary directories."""
    print("=" * 70)
    print("DATASET DIRECTORY SETUP")
    print("=" * 70)
    print()
    
    created = []
    already_exists = []
    
    for dir_path in DIRECTORIES:
        path = Path(dir_path)
        if path.exists():
            already_exists.append(dir_path)
            print(f"  ✓ Already exists: {dir_path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)
            print(f"  ✓ Created: {dir_path}")
    
    print()
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"\nCreated {len(created)} new directories")
    print(f"Found {len(already_exists)} existing directories")
    
    print("\nDirectory structure ready for dataset creation!")
    print("\nNext steps:")
    print("1. Run: python scripts/01_find_ambiguous_pairs.py")
    print("2. Run: python scripts/02_extract_sentences.py")
    print("3. Review and balance the dataset")
    print("4. Generate Baybayin images")


def create_readme():
    """Create README file with instructions."""
    readme_content = """# Baybayin Disambiguation Dataset

## Directory Structure

- **raw/** - Raw extracted sentences from corpora
- **processed/** - Final curated dataset ready for use
- **splits/** - Train/validation/test splits
- **analysis/** - Analysis results and statistics
- **documentation/** - Documentation and annotation guidelines

## Workflow

1. **Discovery**: Find all ambiguous word pairs
   - Script: `01_find_ambiguous_pairs.py`
   - Output: `analysis/ambiguous_pairs_complete.csv`

2. **Extraction**: Mine sentences from corpora
   - Script: `02_extract_sentences.py`
   - Output: `raw/candidate_sentences.json`

3. **Balancing**: Create balanced final dataset
   - Script: `03_balance_dataset.py`
   - Output: `processed/filipino_sentences_v1.txt`

4. **Validation**: Quality checks and statistics
   - Script: `05_validate_dataset.py`
   - Output: `analysis/validation_report.json`

5. **Image Generation**: Create Baybayin images
   - Script: `04_generate_images.py` (adapted from create_dataset.py)
   - Output: `../baybayin_dataset_images/`

## Files

- **filipino_sentences_v1.txt**: Final curated sentences
- **annotations_v1.csv**: Ground truth mappings
- **candidates_results_v1.json**: OCR candidate results
- **dataset_metadata.json**: Complete dataset metadata

## Quality Metrics

- Coverage: All major ambiguous pairs represented
- Balance: Distribution matches target percentages
- Validity: All words exist in Filipino corpus
- Reliability: High inter-annotator agreement
"""
    
    readme_path = Path("dataset/README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  ✓ Created README: {readme_path}")


if __name__ == "__main__":
    create_directories()
    create_readme()
