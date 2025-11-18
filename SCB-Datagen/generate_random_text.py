#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import string
from collections import Counter

# --- Configuration Parameters ---
# Replace with the actual path to your output file
OUTPUT_FILE = "/path/to/your/Synthtext/data/texts_en.txt"
TOTAL_NUM = 1000000  # Total number of text samples to generate

# --- Generation Logic: Unified Weighted Character Pool ---
# This method places all characters (letters, digits, symbols, spaces) into a single pool.
# Their appearance frequency is controlled by weights, allowing for the generation of more natural and random text.
# Post-processing ensures text quality, which is more flexible and powerful than setting multiple generation rules.

# 1. Character Set Definition
LETTERS = string.ascii_letters
DIGITS = string.digits
SYMBOLS = string.punctuation 
SPACE = ' '

# 2. Unified Character Pool
CHARACTER_POOL = LETTERS + DIGITS + SYMBOLS + SPACE

# 3. Character Weight Definition
# Assign weights to each character in the pool to control its generation probability.
# Higher weights mean a higher chance of appearing.
WEIGHTS = (
    [15] * len(LETTERS) +      # Letters have a high frequency
    [10] * len(DIGITS) +       # Digits have a medium frequency
    [1] * len(SYMBOLS) +       # Symbols have a low frequency
    [5] * len(SPACE)           # Spaces have a high individual weight to appear reasonably often
)

# 4. Text Length Definition
LENGTH_WEIGHTS = {
    3: 1, 4: 3, 5: 7, 6: 10, 7: 10, 8: 7,
    9: 3, 10: 1, 11: 1, 12: 1,
}

def generate_text_sample():
    """
    Generates a single text sample by sampling from the unified weighted character pool.
    The function will loop until a valid sample that meets quality standards is generated.
    """
    while True:
        # Step 1: Randomly select a text length
        lengths = list(LENGTH_WEIGHTS.keys())
        weights_len = list(LENGTH_WEIGHTS.values())
        length = random.choices(lengths, weights=weights_len, k=1)[0]

        # Step 2: Randomly draw characters from the weighted pool to form the initial text
        chars = random.choices(CHARACTER_POOL, weights=WEIGHTS, k=length)
        text_sample = "".join(chars)

        # Step 3: Clean and validate the generated text (quality check)
        
        # a) Clean: Remove leading/trailing spaces
        text_sample = text_sample.strip()
        
        # b) Clean: Compress multiple consecutive spaces into one
        text_sample = " ".join(text_sample.split())

        # c) Validate:
        #   - Ensure the text is not empty after cleaning.
        #   - Ensure the text does not consist only of symbols (must contain at least one letter or digit).
        if text_sample and any(c.isalnum() for c in text_sample):
            return text_sample
        # If validation fails, the loop continues to regenerate a sample.

def main():
    print(f"Starting to generate {TOTAL_NUM} random text samples...")
    
    random_texts = []
    length_counter = Counter()

    for i in range(TOTAL_NUM):
        text_sample = generate_text_sample()
        random_texts.append(text_sample)
        length_counter[len(text_sample)] += 1

        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} / {TOTAL_NUM} text samples")

    # Save results
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(random_texts))
        print(f"\nSuccess! Saved {len(random_texts)} text samples to {OUTPUT_FILE}")
        
        # Print final length distribution statistics
        print("\nFinal length distribution statistics:")
        for length in sorted(length_counter.keys()):
            count = length_counter[length]
            percentage = (count / TOTAL_NUM) * 100
            print(f"  Length {length}: {count} times ({percentage:.2f}%)")
            
    except Exception as e:
        print(f"Failed to save: {e}")


if __name__ == "__main__":
    main()
