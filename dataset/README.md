# Baybayin Disambiguation Dataset

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
