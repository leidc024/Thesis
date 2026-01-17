# Import the necessary libraries
from html2image import Html2Image
from PIL import Image
import os
import csv
import base64

# --- CONFIGURATION ---
# You can change these values to match your setup.

# 1. Set the path to your downloaded Baybayin font file.
FONT_PATH = "tagalog stylized.ttf"  # Using Tagalog Stylized font

# 2. Set the desired font size for the images (larger = better quality for OCR).
FONT_SIZE = 150
FONT_WEIGHT = "normal"  # Options: "normal", "bold", "bolder"
LINE_HEIGHT = 1.2  # Adjust vertical spacing between rows (1.0 = tight, 1.5 = normal, 2.0 = double)
WORDS_PER_ROW = 3.0  # Maximum words per row before inserting a line break

# 2.5. Word spacing settings for better OCR recognition
NORMAL_WORD_SPACING = 1.0  # Normal space between words (1.0 = default)
EXTRA_WORD_SPACING = 3.0  # Extra space when transitioning between 1-char and 2-char words

# 3. Set image quality settings for OCR
IMAGE_PADDING = 30  # Padding around text in pixels (top, left, right)
IMAGE_PADDING_BOTTOM = 50  # Extra padding at the bottom in pixels

# 4. Set the names for your input file and output folders/files.
SENTENCES_FILE = "filipino_sentences.txt"
OUTPUT_DIR = "tagalog_stylized_output"
ANNOTATIONS_FILE = "annotations.csv"


# --- SCRIPT START ---

def latin_to_baybayin_tagalog_stylized(text):
    """
    Convert Latin text to Baybayin using Tagalog Stylized font conventions.
    
    Rules for Tagalog Stylized font:
    1. Each consonant = syllable with 'a' vowel (lowercase: b, k, d, g, h, l, m, n, p, s, t, w, y)
    2. Kudlit above (i/e vowel) = type 'i' or 'e' after consonant
    3. Kudlit below (u/o vowel) = type 'u' or 'o' after consonant
    4. Standalone vowels = uppercase A, I, U
    5. nga = uppercase N
    6. Virama (cancel vowel) = + or =
    7. da and ra use same character (d)
    """
    import re
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation (keep letters and spaces)
    text = re.sub(r'[.,!?;:"""''`(){}\[\]<>—–-]', '', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase for processing
    text = text.lower()
    
    vowels = set('aeiou')
    # Include 'r' in consonants - it uses 'd' character in Baybayin
    consonants = set('bkdghlmnprstwy')  # Baybayin consonants
    
    def process_word(word):
        """
        Process word for Tagalog Stylized font:
        - Consonant + 'a' = just the consonant (e.g., 'ba' = 'b')
        - Consonant + 'i/e' = consonant + 'i' (e.g., 'bi' = 'bi')
        - Consonant + 'u/o' = consonant + 'u' (e.g., 'bu' = 'bu')
        - Standalone vowel = uppercase (e.g., 'a' = 'A', 'i' = 'I', 'u' = 'U')
        - Consonant without vowel = consonant + '+' (virama)
        - 'r' is typed as 'd' (they share the same Baybayin character)
        """
        result = []
        i = 0
        
        while i < len(word):
            char = word[i]
            
            # Handle 'ng' digraph (becomes uppercase N)
            if char == 'n' and i + 1 < len(word) and word[i + 1] == 'g':
                # Check if there's a vowel after 'ng'
                if i + 2 < len(word) and word[i + 2] in vowels:
                    vowel = word[i + 2]
                    if vowel == 'a':
                        result.append('N')  # nga = just N
                    elif vowel in 'ie':
                        result.append('Ni')  # ngi/nge = N + kudlit above
                    elif vowel in 'ou':
                        result.append('Nu')  # ngu/ngo = N + kudlit below
                    i += 3
                else:
                    # No vowel after ng, add virama
                    result.append('N+')
                    i += 2
                    
            elif char in consonants:
                # Convert 'r' to 'd' for typing (same Baybayin character)
                typing_char = 'd' if char == 'r' else char
                
                # Check if next character is a vowel
                if i + 1 < len(word) and word[i + 1] in vowels:
                    vowel = word[i + 1]
                    if vowel == 'a':
                        result.append(typing_char)  # Consonant with 'a' = just consonant
                    elif vowel in 'ie':
                        result.append(typing_char + 'i')  # i/e vowel = kudlit above
                    elif vowel in 'ou':
                        result.append(typing_char + 'u')  # u/o vowel = kudlit below
                    i += 2
                else:
                    # Consonant NOT followed by vowel - add virama
                    result.append(typing_char + '+')
                    i += 1
                    
            elif char in vowels:
                # Standalone vowel = uppercase
                if char == 'a':
                    result.append('A')
                elif char in 'ie':
                    result.append('I')
                elif char in 'ou':
                    result.append('U')
                i += 1
            else:
                # Other character
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    def count_baybayin_chars(baybayin_word):
        """Count the number of actual Baybayin characters (excluding kudlit markers)."""
        # Remove kudlit markers (+, i, u) to count base characters
        clean = baybayin_word.replace('+', '').replace('i', '').replace('u', '')
        return len(clean)
    
    # Process each word
    words = text.split(' ')
    processed_words = [process_word(word) for word in words]
    
    # Build the final string with smart spacing
    result_parts = []
    for i, baybayin_word in enumerate(processed_words):
        result_parts.append(baybayin_word)
        
        if i < len(processed_words) - 1:  # Not the last word
            current_char_count = count_baybayin_chars(baybayin_word)
            next_char_count = count_baybayin_chars(processed_words[i + 1])
            
            # Add extra spacing only when transitioning from 1-char to 2-char words
            if current_char_count == 1 and next_char_count == 2:
                result_parts.append(' ' * int(EXTRA_WORD_SPACING))
            else:
                # Normal spacing (including 2-char to 1-char transitions)
                result_parts.append(' ' * int(NORMAL_WORD_SPACING))
    
    processed_line = ''.join(result_parts)
    
    # Group into rows if needed
    if WORDS_PER_ROW and len(processed_words) > WORDS_PER_ROW:
        rows = []
        row_parts = []
        char_count = 0
        
        for i, baybayin_word in enumerate(processed_words):
            row_parts.append(baybayin_word)
            char_count += 1
            
            # Add spacing logic between words in the same row
            if i < len(processed_words) - 1:
                if char_count >= WORDS_PER_ROW:
                    rows.append(''.join(row_parts))
                    row_parts = []
                    char_count = 0
                else:
                    current_char_count = count_baybayin_chars(baybayin_word)
                    next_char_count = count_baybayin_chars(processed_words[i + 1])
                    
                    if (current_char_count == 1 and next_char_count == 2) or \
                       (current_char_count == 2 and next_char_count == 1):
                        row_parts.append(' ' * int(EXTRA_WORD_SPACING))
                    else:
                        row_parts.append(' ' * int(NORMAL_WORD_SPACING))
        
        if row_parts:
            rows.append(''.join(row_parts))
        
        return '\n'.join(rows)
    else:
        return processed_line


def latin_to_baybayin_unicode(text):
    """
    Convert Latin text to actual Unicode Baybayin characters (U+1700-U+171F).
    This is for the CSV annotations, not for font rendering.
    """
    import re
    
    # Clean text
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[.,!?;:"""''`(){}\[\]<>—–-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    result = []
    words = text.split(' ')
    
    for word_idx, word in enumerate(words):
        word_chars = []
        i = 0
        
        while i < len(word):
            char = word[i]
            
            # Handle 'ng' digraph
            if char == 'n' and i + 1 < len(word) and word[i + 1] == 'g':
                if i + 2 < len(word) and word[i + 2] in 'aeiou':
                    vowel = word[i + 2]
                    word_chars.append('\u1709')  # NGA base
                    if vowel in 'ie':
                        word_chars.append('\u1714')  # Kudlit above
                    elif vowel in 'ou':
                        word_chars.append('\u1715')  # Kudlit below
                    i += 3
                else:
                    word_chars.append('\u1709\u1714')  # NGA + virama
                    i += 2
                    
            elif char in 'bkdghlmnprstwy':
                # Map consonant to base character
                base_char = {
                    'b': '\u1703', 'k': '\u1703', 'g': '\u1703',
                    'd': '\u1705', 'r': '\u1705',
                    'l': '\u1706', 'm': '\u1707',
                    'n': '\u1708', 'p': '\u170A',
                    's': '\u170B', 't': '\u170C',
                    'w': '\u170D', 'y': '\u170E',
                    'h': '\u170F'
                }.get(char, char)
                
                # Check for following vowel
                if i + 1 < len(word) and word[i + 1] in 'aeiou':
                    vowel = word[i + 1]
                    word_chars.append(base_char)
                    if vowel in 'ie':
                        word_chars.append('\u1714')  # Kudlit above
                    elif vowel in 'ou':
                        word_chars.append('\u1715')  # Kudlit below
                    # 'a' needs no kudlit (inherent)
                    i += 2
                else:
                    word_chars.append(base_char + '\u1714')  # Add virama
                    i += 1
                    
            elif char in 'aeiou':
                # Standalone vowel
                if char == 'a':
                    word_chars.append('\u1700')
                elif char in 'ie':
                    word_chars.append('\u1701')
                elif char in 'ou':
                    word_chars.append('\u1702')
                i += 1
            else:
                i += 1
        
        result.append(''.join(word_chars))
        
        # Add spacing between words (every WORDS_PER_ROW words, add newline)
        if (word_idx + 1) % WORDS_PER_ROW == 0 and word_idx < len(words) - 1:
            result.append('\n')
        elif word_idx < len(words) - 1:
            result.append('   ')
    
    return ''.join(result)


def generate_images():
    """
    Reads sentences from a text file, converts each into a Baybayin image
    using Tagalog Stylized font, and creates optimally-sized images.
    """
    # --- 1. SETUP ---
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Check if font file exists
    if not os.path.exists(FONT_PATH):
        print(f"--- ERROR ---")
        print(f"Font file '{FONT_PATH}' not found.")
        print("Please make sure 'tagalog stylized.ttf' is in the same folder as this script.")
        return

    # Initialize Html2Image for rendering text with proper font support
    hti = Html2Image(output_path=OUTPUT_DIR)
    
    print(f"Using font: {FONT_PATH}")

    # --- 2. PROCESSING ---
    annotations = []
    print("\nStarting image generation with Tagalog Stylized font...")
    
    sentences_found = False

    # Open the file containing your Filipino sentences
    try:
        with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                sentence = line.strip()
                if not sentence:
                    continue
                
                sentences_found = True

                # Convert Latin text to Baybayin (for font rendering)
                baybayin_text = latin_to_baybayin_tagalog_stylized(sentence)
                # Convert to Unicode Baybayin (for CSV)
                baybayin_unicode = latin_to_baybayin_unicode(sentence)
                
                print(f"Processing sentence {i+1}: {sentence[:50]}...")
                print(f"  Baybayin: {baybayin_text[:50]}...")
                print(f"  Unicode: {baybayin_unicode[:50]}...")

                # --- Create the image for each sentence using HTML/CSS rendering ---
                
                # Embed font as base64 in HTML
                with open(FONT_PATH, 'rb') as font_file:
                    font_base64 = base64.b64encode(font_file.read()).decode('utf-8')
                
                # Determine font format for CSS
                font_format = 'truetype'
                
                # Use unique font name to avoid caching issues
                unique_font_name = f'TagalogStylized_{i}'
                
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
                            text-rendering: optimizeLegibility;
                            letter-spacing: 2px;  /* Slight letter spacing for clarity */
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
                    # Add the configured padding (extra padding on bottom)
                    bbox = (
                        max(0, bbox[0] - IMAGE_PADDING),
                        max(0, bbox[1] - IMAGE_PADDING),
                        min(white_bg.width, bbox[2] + IMAGE_PADDING),
                        min(white_bg.height, bbox[3] + IMAGE_PADDING_BOTTOM)  # Extra bottom padding
                    )
                    white_bg = white_bg.crop(bbox)
                else:
                    # If no text found, just add padding to original
                    print(f"Warning: No text found in image {i+1}, using original size")
                
                # Save with high quality
                white_bg.save(img_path, 'PNG', optimize=False, quality=100)
                
                print(f"  Generated: {image_filename} ({white_bg.width}x{white_bg.height})")
                
                # Store the mapping for annotations (ground truth)
                # Use Unicode Baybayin characters in CSV, not the typing representation
                annotations.append([image_filename, baybayin_unicode, sentence])
                print()

    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"Input file '{SENTENCES_FILE}' not found.")
        print("Please create this file and add your Filipino sentences to it, one per line.")
        return
    
    # --- 3. CREATE THE ANNOTATIONS CSV (Your Ground Truth File) ---
    if not sentences_found:
        print("No sentences were processed. Skipping CSV creation.")
        return
    
    print(f"\n{'='*60}")
    print("Creating annotations CSV file...")
    print(f"Annotations count: {len(annotations)}")
    
    annotations_path = os.path.join(OUTPUT_DIR, ANNOTATIONS_FILE)
    print(f"CSV path: {annotations_path}")
    
    try:
        with open(annotations_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["filename", "baybayin_text", "latin_text"])
            writer.writerows(annotations)
        print(f"✓ Annotations CSV created successfully!")
        print(f"✓ File location: {os.path.abspath(annotations_path)}")
    except Exception as e:
        print(f"ERROR creating CSV: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Image generation complete!")
    print(f"-> Images saved in the '{OUTPUT_DIR}' folder.")
    print(f"-> Total images generated: {len(annotations)}")
    print(f"-> Ground truth mapping saved in '{annotations_path}'.")
    print(f"{'='*60}")


# --- Run the main function ---
if __name__ == "__main__":
    generate_images()