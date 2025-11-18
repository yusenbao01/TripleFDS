#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw, ImageFont
import glob
from pathlib import Path

# Configuration
# Replace with the actual path to your font directory
FONT_DIR = "/path/to/your/font_directory"
OUTPUT_DIR = "./test_results"  # Output directory
ENGLISH_TEXT = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ   !\"#$%&'()*+,-./:;<=>?@[]^_`{|}~"
IMAGE_WIDTH = 3000
IMAGE_HEIGHT = 300
FONT_SIZE = 40
BG_COLOR = (255, 255, 255)  # White
TEXT_COLOR = (0, 0, 0)  # Black


def create_font_test_image(font_path, output_dir):
    """Creates a test image for the specified font."""
    font_name = os.path.basename(font_path).rsplit(".", 1)[0]

    try:
        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=BG_COLOR)
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(font_path, FONT_SIZE)

        # Draw title - display font name
        title_font = ImageFont.truetype(font_path, FONT_SIZE // 2)
        draw.text((20, 20), f"Font: {font_name}", fill=(100, 100, 100), font=title_font)

        # Draw English text
        draw.text((50, 80), ENGLISH_TEXT, fill=TEXT_COLOR, font=font)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{font_name}.png")
        img.save(output_path)

        print(f"✓ Generated: {font_name}")
        return True
    except Exception as e:
        print(f"✗ Error ({font_name}): {str(e)}")
        # Create an error image
        try:
            img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(255, 240, 240))
            draw = ImageDraw.Draw(img)
            # Use a system default font if possible, otherwise this might fail too
            draw.text((20, 20), f"Font: {font_name} (Failed to load)", fill=(255, 0, 0))
            draw.text((20, 80), f"Error: {str(e)}", fill=(200, 0, 0))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{font_name}_error.png")
            img.save(output_path)
        except:
            pass
        return False


def test_all_fonts():
    """Tests all TTF and OTF fonts in the directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all TTF and OTF font files
    font_files = []
    for ext in ["*.ttf", "*.otf", "*.TTF", "*.OTF"]:
        font_files.extend(glob.glob(os.path.join(FONT_DIR, ext)))

    print(f"Found {len(font_files)} font files in total.")

    success_count = 0
    failure_count = 0

    for font_path in font_files:
        if create_font_test_image(font_path, OUTPUT_DIR):
            success_count += 1
        else:
            failure_count += 1

    print("\nTesting finished!")
    print(f"Success: {success_count}")
    print(f"Failure: {failure_count}")
    print(f"Total: {success_count + failure_count}")
    print(f"\nResults saved in: {OUTPUT_DIR}")

    create_summary_image(OUTPUT_DIR)


def create_summary_image(output_dir):
    """Creates a summary image of all fonts."""
    try:
        # Get all generated images
        image_files = [f for f in os.listdir(output_dir) if f.endswith(".png") and not f.endswith("_summary.png")]

        if not image_files:
            print("No image files found, cannot create summary.")
            return

        # Calculate summary image dimensions
        cols = 3
        rows = (len(image_files) + cols - 1) // cols

        # Create a blank summary image
        summary_img = Image.new("RGB", (cols * IMAGE_WIDTH // 2, rows * IMAGE_HEIGHT // 2), color=(240, 240, 240))

        # Arrange all font images
        for idx, img_file in enumerate(sorted(image_files)):
            try:
                img_path = os.path.join(output_dir, img_file)
                img = Image.open(img_path)
                img = img.resize((IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2))

                x = (idx % cols) * (IMAGE_WIDTH // 2)
                y = (idx // cols) * (IMAGE_HEIGHT // 2)

                summary_img.paste(img, (x, y))
            except Exception as e:
                print(f"Error adding image to summary {img_file}: {e}")

        # Save the summary image
        summary_path = os.path.join(output_dir, "..", "fonts_summary.png")
        summary_img.save(summary_path)
        print(f"Summary image created: {summary_path}")
    except Exception as e:
        print(f"Error creating summary image: {e}")


if __name__ == "__main__":
    test_all_fonts()
