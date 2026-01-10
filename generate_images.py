# Import the necessary libraries
from html2image import Html2Image
from PIL import Image
import os
import csv
import base64

# --- CONFIGURATION ---
# You can change these values to match your setup.

# 1. Set the path to your downloaded Baybayin font file.
FONT_PATH = "BaybayinNamin.otf"  # Using BaybayinNamin font

# 2. Set the desired font size for the images (larger = better quality for OCR).
FONT_SIZE = 150
FONT_WEIGHT = "normal"  # Options: "normal", "bold", "bolder"
LINE_HEIGHT = 1.2  # Adjust vertical spacing between rows (1.0 = tight, 1.5 = normal, 2.0 = double)
WORDS_PER_ROW = 3  # Maximum words per row before inserting a line break

# 3. Set image quality settings for OCR
IMAGE_PADDING = 30  # Padding around text in pixels

# 4. Set the names for your input file and output folders/files.
SENTENCES_FILE = "filipino_sentences.txt"
OUTPUT_DIR = "try_new_font"


# --- SCRIPT START ---

def show_unicode(text):
    """Helper function to display Unicode code points of characters."""
    return ' '.join(f'U+{ord(c):04X}' for c in text)

def latin_to_baybayin(text):
    """
    For BaybayinNamin.otf font: This font maps Latin letters directly to Baybayin glyphs
    using OpenType ligatures. 
    
    The font uses '=' to trigger krus-kudlit (virama) for consonants without vowels.
    Ligatures: consonant + UPPERCASE vowel (e.g., 'rI' for ri with i-kudlit)
    """
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation (keep letters and spaces)
    text = re.sub(r'[.,!?;:"""''`(){}\[\]<>—–-]', '', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # For BaybayinNamin font: use lowercase Latin letters
    text = text.lower()
    
    # Process each word to add krus-kudlit (=) for consonants without following vowels
    vowels = set('aeiou')
    consonants = set('bdfghjklmnprstvwy')  # Baybayin consonants
    
    def process_word(word):
        """
        Process word for BaybayinNamin font:
        - Consonant + vowel = consonant (lowercase) + VOWEL (UPPERCASE) for ligature
        - Consonant alone = add '=' for krus-kudlit
        - Standalone vowel at start = keep as lowercase
        """
        result = []
        i = 0
        while i < len(word):
            char = word[i]
            
            # Handle 'ng' digraph
            if char == 'n' and i + 1 < len(word) and word[i + 1] == 'g':
                # Check if there's a vowel after 'ng'
                if i + 2 < len(word) and word[i + 2] in vowels:
                    # ng + vowel: output ng + UPPERCASE vowel for ligature
                    result.append('ng' + word[i + 2].upper())
                    i += 3
                else:
                    # No vowel after ng, add krus-kudlit
                    result.append('ng=')
                    i += 2
            elif char in consonants:
                # Check if next character is a vowel
                if i + 1 < len(word) and word[i + 1] in vowels:
                    # Consonant + vowel: output consonant + UPPERCASE vowel for ligature
                    result.append(char + word[i + 1].upper())
                    i += 2  # Skip both consonant and vowel
                else:
                    # Consonant NOT followed by vowel - add krus-kudlit
                    result.append(char + '=')
                    i += 1
            elif char in vowels:
                # Standalone vowel (at word start or after another vowel)
                result.append(char)
                i += 1
            else:
                # Other character (shouldn't happen, but keep it)
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    # Process each word
    words = text.split(' ')
    processed_words = [process_word(word) for word in words]
    
    # Group words into rows of WORDS_PER_ROW
    rows = []
    for i in range(0, len(processed_words), WORDS_PER_ROW):
        row_words = processed_words[i:i + WORDS_PER_ROW]
        rows.append('   '.join(row_words))  # 6 spaces between words
    
    return '\n'.join(rows)  # Join rows with newlines

def generate_images():
    """
    Reads sentences from a text file, converts each into a Baybayin image
    using BaybayinNamin.otf font, and creates a CSV file mapping images 
    to their ground-truth text.
    """
    # --- 1. SETUP ---
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Check if font file exists
    if not os.path.exists(FONT_PATH):
        print(f"--- ERROR ---")
        print(f"Font file '{FONT_PATH}' not found.")
        print("Please make sure BaybayinNamin.otf is in the same folder as this script.")
        return

    # Initialize Html2Image for rendering text with proper font support
    hti = Html2Image(output_path=OUTPUT_DIR)
    
    print(f"Using font: {FONT_PATH}")

    # --- 2. PROCESSING ---
    annotations = []
    print("\nStarting image generation with BaybayinNamin font...")

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

                # --- Create the image for each sentence using HTML/CSS rendering ---
                
                # Embed font as base64 in HTML (works for both .otf and .ttf)
                with open(FONT_PATH, 'rb') as font_file:
                    font_base64 = base64.b64encode(font_file.read()).decode('utf-8')
                
                # Determine font format for CSS
                font_format = 'opentype' if FONT_PATH.endswith('.otf') else 'truetype'
                
                # Use unique font name to avoid caching issues
                unique_font_name = f'BaybayinCustom_{i}'
                
                # Create HTML with embedded font - use large canvas first
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        @font-face {{
                            font-family: '{unique_font_name}';
                            src: url(data:font/{font_format};charset=utf-8;base64,{font_base64}) format('{font_format}');
                            font-display: block;
                        }}
                        html, body {{
                            margin: 0;
                            padding: {IMAGE_PADDING}px;
                            background-color: white !important;
                            background: white !important;
                        }}
                        .baybayin {{
                            font-family: '{unique_font_name}' !important;
                            font-size: {FONT_SIZE}px;
                            font-weight: {FONT_WEIGHT};
                            line-height: {LINE_HEIGHT};
                            color: black;
                            white-space: pre-wrap;
                            background-color: white;
                            display: inline-block;
                            /* Enable OpenType ligatures for proper Baybayin rendering */
                            font-feature-settings: "liga" 1, "clig" 1, "dlig" 1, "calt" 1;
                            -webkit-font-feature-settings: "liga" 1, "clig" 1, "dlig" 1, "calt" 1;
                            font-variant-ligatures: common-ligatures discretionary-ligatures contextual;
                            text-rendering: optimizeLegibility;
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
                
                # Render HTML to image with large initial size
                hti.screenshot(
                    html_str=html_content,
                    save_as=image_filename,
                    size=(3000, 2000)  # Large canvas to capture everything
                )
                
                # Post-process: Crop to exact text boundaries
                img_path = os.path.join(OUTPUT_DIR, image_filename)
                img = Image.open(img_path).convert('RGBA')
                
                # Create white background
                white_bg = Image.new('RGB', img.size, 'white')
                white_bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                
                # Crop to remove all whitespace, keeping only the text
                gray = white_bg.convert('L')
                # Find bounding box of non-white pixels
                bbox = gray.point(lambda x: 0 if x > 250 else 255).getbbox()
                
                if bbox:
                    # Add the configured padding
                    bbox = (
                        max(0, bbox[0] - IMAGE_PADDING),
                        max(0, bbox[1] - IMAGE_PADDING),
                        min(white_bg.width, bbox[2] + IMAGE_PADDING),
                        min(white_bg.height, bbox[3] + IMAGE_PADDING)
                    )
                    white_bg = white_bg.crop(bbox)
                else:
                    # If no text found, just add padding to original
                    print(f"Warning: No text found in image {i+1}, using original size")
                
                # Save with high quality
                white_bg.save(img_path, 'PNG', optimize=False, quality=100)
                
                print(f"Generated: {image_filename} ({white_bg.width}x{white_bg.height})")
                
                # Store the mapping for our annotations file (the ground truth)
                annotations.append([image_filename, baybayin_text, sentence])
                print()

    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"Input file '{SENTENCES_FILE}' not found.")
        print("Please create this file and add your Filipino sentences to it, one per line.")
        return

    # # --- 3. CREATE THE ANNOTATIONS CSV (Your Ground Truth File) ---
    # with open(ANNOTATIONS_FILE, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig adds BOM for Excel
    #     writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields to handle special characters
    #     writer.writerow(["filename", "baybayin_text", "latin_text"])  # Write the header
    #     writer.writerows(annotations)

    print(f"\nImage generation complete!")
    print(f"-> Images saved in the '{OUTPUT_DIR}' folder.")
    # print(f"-> Ground truth mapping saved in '{ANNOTATIONS_FILE}'.")


# --- Run the main function ---
if __name__ == "__main__":
    generate_images()