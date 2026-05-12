# MaBaybay-OCR 2.0: Context-Aware Baybayin Transliteration

MaBaybay-OCR 2.0 addresses the inherent phonetic ambiguities of the Baybayin script (specifically e/i, o/u, and d/r overlaps) by applying a bidirectional Masked Language Model (MLM) scoring mechanism of Salazar et.al 2021. By leveraging jcblaise/roberta-tagalog-base, the system achieves a disambiguation accuracy of **83%**, significantly outperforming the 46% baseline of the OCR's heuristic-based systems.

## 1. Prerequisites

### Software Requirements

- **MATLAB**: Version R2023b or later
- **Python**: Version 3.8 to 3.11 (required for the NLP disambiguation engine)

### Required MATLAB Toolboxes

Ensure the following toolboxes are installed via the MATLAB Add-On Explorer:

- Statistics and Machine Learning Toolbox
- Image Processing Toolbox
- Computer Vision Toolbox
- OCR Language Data Files (Tagalog) (Essential for Tagalog character support)
- Parallel Computing Toolbox

## 2. Installation and Setup

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/leidc024/Thesis
cd Thesis
```

### Step 2: Python Environment Setup

**Option A (Recommended): Using Virtual Environment**

Create a virtual environment to isolate dependencies and avoid conflicts:

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

**Option B: Direct Installation (Without Virtual Environment)**

If you prefer to skip the virtual environment:

```bash
pip install -r requirements.txt
```

### Step 3: Model and Corpus Initialization

The system uses the `jcblaise/roberta-tagalog-base` model from Hugging Face. Upon the first execution, the system will automatically download the necessary weights.

Ensure your `corpus/` folder contains the following files for frequency-based scoring:

- `Tagalog_Literary_Text.txt`
- `Tagalog_Religious_Text.txt`
- `Tagalog_Balita_Texts_Balanced.txt`

## 3. System Integration

The system operates via a hybrid bridge. MATLAB handles image segmentation and character recognition, while Python handles semantic rescoring.

- **MATLAB Side**: The script `disambiguate_candidates.m` captures OCR candidates and exports them to a temporary `candidates.json` file.
- **Python Side**: MATLAB triggers `disambiguate.py` via a system call.
- **Scoring**: The `BaybayinDisambiguator` class computes Pseudo-Log-Likelihood (PLL) scores:

$$\text{PLL(candidate)} = \sum \log P(\text{tokens} \mid \text{context})$$

- **Output**: The disambiguated text is returned to the MATLAB terminal for final display.

## 4. Running the System

### Standard Usage

1. Open MATLAB and navigate to the project folder.
2. Run the main GUI or script: `MaBaybay_GUI.mlapp`.
3. Upload an image of Baybayin text.
4. Click **Transliterate**. The terminal will show the "context-aware" selection process.

### Running the Testing Framework

To reproduce the research results (83% accuracy), run the unit tests provided in the `tests/` directory:

```bash
# Ensure venv is active
python tests/test_asero_asido.py
python tests/test_bote_buti.py
```

The results and timing logs will be saved in `gold_standard_dataset/results/` in JSON format.

## 5. Project Structure

- **src/**: Contains `disambiguator.py` (Core scoring logic)
- **gold_standard_dataset/**: The 1,600-sample benchmark dataset
  - **sentences/**: Ground truth text files
  - **images/**: Handwritten Baybayin test images
- **disambiguate.py**: The entry point wrapper for MATLAB system calls
- **ambiguous_pairs_complete.csv**: The mapping of 74,419+ words used for candidate generation

## 6. Acknowledgments

This study was developed under the guidance of Sir Roxas and builds upon the pioneering work of Pino, Mendoza, and Sambayan (2025).