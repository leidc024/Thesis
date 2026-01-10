# Import the necessary libraries
from html2image import Html2Image
from PIL import Image
import os
import csv
import base64

# --- CONFIGURATION ---
# You can change these values to match your setup.

# 1. Set the path to your downloaded Baybayin font file.
#    Make sure this file is in the same folder as your script.
FONT_PATH = "NotoSansTagalog.ttf"  # <-- IMPORTANT: CHANGE THIS to your font file name

# 2. Set the desired font size for the images (larger = better quality for OCR).
FONT_SIZE = 80
FONT_WEIGHT = "bold"  # Options: "normal", "bold", "bolder" - Bold improves OCR recognition

# 3. Set image quality settings for OCR
IMAGE_DPI = 300  # High DPI for print quality
IMAGE_WIDTH = 1200  # Width in pixels
IMAGE_HEIGHT = 800  # Height in pixels (will auto-adjust based on text)

# 4. Set the names for your input file and output folders/files.
SENTENCES_FILE = "dataset/processed/test_sentences_500.txt"
OUTPUT_DIR = "baybayin_dataset_images"
ANNOTATIONS_FILE = "dataset/processed/ground_truth.csv"

# --- SCRIPT START ---

def show_unicode(text):
    """Helper function to display Unicode code points of characters."""
    return ' '.join(f'U+{ord(c):04X}' for c in text)

def latin_to_baybayin(text):
    """
    Converts Latin script Filipino text to Baybayin script.
    This is a basic transliteration based on phonetic rules.
    Removes punctuation and special characters before conversion.
    """
    # Remove punctuation and special characters
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation (keep letters, spaces, and apostrophes in contractions)
    text = re.sub(r'[.,!?;:"""''`(){}\[\]<>—–-]', '', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Baybayin Unicode character mappings
    baybayin_chars = {
        # Independent vowels
        'a': 'ᜀ',
        'e': 'ᜁ',
        'i': 'ᜁ',
        'o': 'ᜂ',
        'u': 'ᜂ',
        
        # Consonants (inherent 'a' sound)
        'ka': 'ᜃ', 'ga': 'ᜄ', 'nga': 'ᜅ',
        'ta': 'ᜆ', 'da': 'ᜇ','ra': 'ᜇ', 'na': 'ᜈ',
        'pa': 'ᜉ', 'ba': 'ᜊ', 'ma': 'ᜋ',
        'ya': 'ᜌ', 'la': 'ᜎ', 'wa': 'ᜏ',
        'sa': 'ᜐ', 'ha': 'ᜑ',
        
        # Consonants with vowel marks
        'ki': 'ᜃᜒ', 'ke': 'ᜃᜒ', 'ku': 'ᜃᜓ', 'ko': 'ᜃᜓ',
        'gi': 'ᜄᜒ', 'ge': 'ᜄᜒ', 'gu': 'ᜄᜓ', 'go': 'ᜄᜓ',
        'ngi': 'ᜅᜒ', 'nge': 'ᜅᜒ', 'ngu': 'ᜅᜓ', 'ngo': 'ᜅᜓ',
        'ti': 'ᜆᜒ', 'te': 'ᜆᜒ', 'tu': 'ᜆᜓ', 'to': 'ᜆᜓ',
        'di': 'ᜇᜒ', 'de': 'ᜇᜒ', 'du': 'ᜇᜓ', 'do': 'ᜇᜓ',
        'ri': 'ᜇᜒ', 're': 'ᜇᜒ', 'ru': 'ᜇᜓ', 'ro': 'ᜇᜓ',
        'ni': 'ᜈᜒ', 'ne': 'ᜈᜒ', 'nu': 'ᜈᜓ', 'no': 'ᜈᜓ',
        'pi': 'ᜉᜒ', 'pe': 'ᜉᜒ', 'pu': 'ᜉᜓ', 'po': 'ᜉᜓ',
        'bi': 'ᜊᜒ', 'be': 'ᜊᜒ', 'bu': 'ᜊᜓ', 'bo': 'ᜊᜓ',
        'mi': 'ᜋᜒ', 'me': 'ᜋᜒ', 'mu': 'ᜋᜓ', 'mo': 'ᜋᜓ',
        'yi': 'ᜌᜒ', 'ye': 'ᜌᜒ', 'yu': 'ᜌᜓ', 'yo': 'ᜌᜓ',
        'li': 'ᜎᜒ', 'le': 'ᜎᜒ', 'lu': 'ᜎᜓ', 'lo': 'ᜎᜓ',
        'wi': 'ᜏᜒ', 'we': 'ᜏᜒ', 'wu': 'ᜏᜓ', 'wo': 'ᜏᜓ',
        'si': 'ᜐᜒ', 'se': 'ᜐᜒ', 'su': 'ᜐᜓ', 'so': 'ᜐᜓ',
        'hi': 'ᜑᜒ', 'he': 'ᜑᜒ', 'hu': 'ᜑᜓ', 'ho': 'ᜑᜓ',
        
        # Kundlit / Krus-kudlit (cancels inherent vowel)
        'k': 'ᜃ᜔', 'g': 'ᜄ᜔', 'ng': 'ᜅ᜔',
        't': 'ᜆ᜔', 'd': 'ᜇ᜔', 'r': 'ᜇ᜔', 'n': 'ᜈ᜔',
        'p': 'ᜉ᜔', 'b': 'ᜊ᜔', 'm': 'ᜋ᜔',
        'y': 'ᜌ᜔', 'l': 'ᜎ᜔', 'w': 'ᜏ᜔',
        's': 'ᜐ᜔', 'h': 'ᜑ᜔',
    }
    
    text = text.lower()
    result = []
    i = 0
    
    while i < len(text):
        # Skip spaces and add extra spacing between words
        if text[i] == ' ':
            result.append('      ')  # Add 7 spaces instead of 1 for better word separation
            i += 1
            continue
        
        # Try to match longer patterns first (3 chars, 2 chars, 1 char)
        matched = False
        
        # Try 3-character match (for 'nga', 'ngi', etc.)
        if i + 2 < len(text):
            three_char = text[i:i+3]
            if three_char in baybayin_chars:
                result.append(baybayin_chars[three_char])
                i += 3
                matched = True
                continue
        
        # Try 2-character match
        if i + 1 < len(text):
            two_char = text[i:i+2]
            if two_char in baybayin_chars:
                result.append(baybayin_chars[two_char])
                i += 2
                matched = True
                continue
        
        # Try 1-character match
        one_char = text[i]
        if one_char in baybayin_chars:
            result.append(baybayin_chars[one_char])
            i += 1
            matched = True
        else:
            # If no match, keep the original character
            result.append(one_char)
            i += 1
    
    # Post-process: Add line breaks before single-character words
    baybayin_text = ''.join(result)
    words = baybayin_text.split('   ')  # Split by our 3-space separator
    processed_words = []
    
    for word in words:
        # Count actual Baybayin characters (exclude spaces and diacritics)
        char_count = 0
        for char in word:
            if '\u1700' <= char <= '\u171F' and char != '\u1714':  # Baybayin range, exclude virama
                if char < '\u1712':  # Not a diacritic mark
                    char_count += 1
        
        # If word is a single character, add newline before it (unless it's the first word)
        if char_count == 1 and processed_words:
            processed_words.append('\n' + word)
        else:
            processed_words.append(word)
    
    return '   '.join(processed_words)

def generate_dataset():
    """
    Reads sentences from a text file, converts each into a Baybayin image,
    and creates a CSV file mapping images to their ground-truth text.
    """
    # --- 1. SETUP ---
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize Html2Image for rendering text with proper font support
    hti = Html2Image(output_path=OUTPUT_DIR)
    
    print(f"Using font: {FONT_PATH}")

    # --- 2. PROCESSING ---
    annotations = []
    print("\nStarting dataset generation...")

    # Open the file containing your Filipino sentences
    try:
        with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                sentence = line.strip()
                if not sentence:
                    continue

                # Convert Latin text to Baybayin
                baybayin_text = latin_to_baybayin(sentence)
                print(f"Processing sentence {i+1}: {sentence[:50]}...")
                # print(f"Baybayin: {baybayin_text}")  # Disabled due to console encoding issues
                # print(f"Unicode:  {show_unicode(baybayin_text)}")


                # --- Create the image for each sentence using HTML/CSS rendering ---
                
                # Embed font as base64 in HTML
                with open(FONT_PATH, 'rb') as font_file:
                    font_base64 = base64.b64encode(font_file.read()).decode('utf-8')
                
                # Create HTML with embedded font
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        @font-face {{
                            font-family: 'BaybayinFont';
                            src: url(data:font/truetype;charset=utf-8;base64,{font_base64}) format('truetype');
                        }}
                        html, body {{
                            margin: 0;
                            padding: 20px;
                            background-color: white !important;
                            background: white !important;
                        }}
                        .baybayin {{
                            font-family: 'BaybayinFont', sans-serif;
                            font-size: {FONT_SIZE}px;
                            font-weight: {FONT_WEIGHT};
                            color: black;
                            white-space: pre-wrap;
                            background-color: white;
                            
                        }}
                    </style>
                </head>
                <body>
                    <div class="baybayin">{baybayin_text}</div>
                </body>
                </html>
                """
                
                # Generate image filename
                image_filename = f"sentence_{i+1}.png"
                
                # Render HTML to image using browser engine with high quality settings
                hti.screenshot(
                    html_str=html_content,
                    save_as=image_filename,
                    size=(IMAGE_WIDTH, IMAGE_HEIGHT)
                )
                
                # Post-process: Ensure white background and optimize for OCR
                img_path = os.path.join(OUTPUT_DIR, image_filename)
                img = Image.open(img_path).convert('RGBA')
                
                # Create white background
                white_bg = Image.new('RGB', img.size, 'white')
                white_bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                
                # Crop to remove excess whitespace (improves OCR)
                # Convert to grayscale to find text boundaries
                gray = white_bg.convert('L')
                bbox = gray.getbbox()
                if bbox:
                    # Add padding back
                    padding = 20
                    bbox = (
                        max(0, bbox[0] - padding),
                        max(0, bbox[1] - padding),
                        min(white_bg.width, bbox[2] + padding),
                        min(white_bg.height, bbox[3] + padding)
                    )
                    white_bg = white_bg.crop(bbox)
                
                # Save with high quality
                white_bg.save(img_path, 'PNG', optimize=False, quality=100)
                
                print(f"Generated: {image_filename} ({white_bg.width}x{white_bg.height})")
                
                # E. Store the mapping for our annotations file (the ground truth)
                annotations.append([image_filename, baybayin_text, sentence])
                print()

    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"Input file '{SENTENCES_FILE}' not found.")
        print("Please create this file and add your Filipino sentences to it, one per line.")
        return

    # --- 3. CREATE THE ANNOTATIONS CSV (Your Ground Truth File) ---
    with open(ANNOTATIONS_FILE, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig adds BOM for Excel
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields to handle special characters
        writer.writerow(["filename", "baybayin_text", "latin_text"]) # Write the header
        writer.writerows(annotations)

    print(f"\nDataset generation complete!")
    print(f"-> Images saved in the '{OUTPUT_DIR}' folder.")
    print(f"-> Ground truth mapping saved in '{ANNOTATIONS_FILE}'.")


# --- Run the main function ---
if __name__ == "__main__":
    generate_dataset()