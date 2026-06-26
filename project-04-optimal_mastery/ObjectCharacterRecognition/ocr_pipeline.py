"""
Project 4 - Path 1: OCR (Optical Character Recognition)
DecodeLabs | Batch 2026
Tool: pytesseract (Google Tesseract Engine)

"""

import cv2 # This imports OpenCV (Open Source Computer Vision Library) .It handles all the image processing tasks.
import pytesseract 
# which is a Python wrapper for Google's Tesseract OCR (Optical Character Recognition) engine.
import numpy as np
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_image(image_path):
    """Load image from disk."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    print(f"[✓] Image loaded: {image_path} | Shape: {img.shape}")
    return img


def preprocess_image(img):
    """
    Pre-process image to clean it for better OCR accuracy.
    Steps:
      1. Grayscale - Remove color, collapse RGB to 1 channel
      2. Gaussian Blur - Remove noise/artifacts
      3. Adaptive Thresholding - Convert to pure black & white
    """

    # Step 1: Grayscale conversion

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("[✓] Step 1: Grayscale conversion done")

    # Step 2: Gaussian Blur to remove noise

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("[✓] Step 2: Gaussian Blur applied")

    # (5, 5): This is the Kernel Size (Width x Height). Think of it as a small $5 \times 5$ pixel window that slides across your entire image to average out pixel values.
    # 0: This represents the standard deviation in the X direction (sigma x).

     # Step 3: Adaptive Thresholding (Otsu's Method) 
    # Forces every pixel to 0 (black) or 255 (white)


    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # implementation of Otsu's Automatic Thresholding.
    # _ (Underscore): This is a dummy variable placeholder. The cv2.threshold function returns two values: the optimal threshold limit that it calculated, and the modified image itself. Since you don't need to use the exact cutoff number later, Python convention uses _ to discard it.
    # thresh: This is the final variable where your newly generated, high-contrast binary (black and white) image matrix is saved.
    # cv2.threshold(...): The core OpenCV function used to separate pixels based on light intensity.
    # 0: The initial threshold value.
    # 255: The maximum value to assign. If a pixel is determined to be lighter than the calculated threshold, it gets maxed out to 255 (pure white). Anything darker drops to 0 (pure black).
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU: These are flags combined using a + sign:
    # cv2.THRESH_BINARY: Tells OpenCV to output standard binary format (only absolute black or absolute white).
    # cv2.THRESH_OTSU: Crucial flag. It triggers Otsu’s algorithm, which analyzes the image's pixel histogram, finds the distinct peaks of the foreground (text) and background, and calculates the optimal contrast split point completely on its own.

    print("[✓] Step 3: Adaptive Thresholding (Otsu's) applied")
 
    return thresh


def run_ocr(preprocessed_img, psm_mode=6):
    """
    Run Tesseract OCR on the preprocessed image.
    PSM Modes:
      3  = Fully automatic (default)
      6  = Single block of text (book/document page)
      7  = Single text line (number plates, headers)
      11 = Sparse/scattered text (invoices, receipts)
    """

    config = f'--oem 3 --psm {psm_mode}'
    text = pytesseract.image_to_string(preprocessed_img, config=config)
    return text.strip()

# OEM stands for OCR Engine Mode. It tells Tesseract which underlying mathematical model it should use to recognize characters.
# It acts as the bridge that sends your image data over to Google's Tesseract OCR engine, tells it to process the image, and converts the engine's raw output into a clean string.


def get_confidence(preprocessed_img, psm_mode=6):
    """
    Get word-level confidence scores from Tesseract.
    Returns average confidence of recognized words.
    """
    config = f'--oem 3 --psm {psm_mode}'
    data = pytesseract.image_to_data(preprocessed_img, config=config, output_type=pytesseract.Output.DICT)

    confidences = [int(c) for c in data['conf'] if int(c) != -1]
    if not confidences:
        return 0
    avg_conf = sum(confidences) / len(confidences)
    return avg_conf
 
 # data['conf']: Tesseract analyzes the image word-by-word and generates a raw list of confidence ratings (from 0 to 100) for each word it thinks it found.
 # if int(c) != -1: When Tesseract detects structural spaces, newlines, or blank blocks instead of actual words, it assigns them a placeholder value of -1. This condition filters out all those -1 values so they don't break your math.
 # int(c): Converts the valid scores from text strings into integers so you can perform calculations on them.
 # Result: You get a clean list containing only valid, numerical accuracy scores (e.g., [85, 90, 78, 92])

def save_preprocessed_image(preprocessed_img, output_path="preprocessed_output.jpg"):
    """Save the pre-processed image so you can visually verify it."""
    cv2.imwrite(output_path, preprocessed_img)
    print(f"[✓] Preprocessed image saved: {output_path}")

    # cv2.imwrite(): This is the OpenCV function used to write/save an image matrix to a physical file on disk.
    # It takes the target filename string (output_path) and the image array data (preprocessed_img), compresses it into a standard image format, and saves it in your project folder.


def main():
    # -------------------------------------------------------
    # INPUT: Put your image filename here
    IMAGE_PATH = "sample_text.jpg"   # Change this to your image file
    PSM_MODE = 6                      # Change based on your image type
    CONFIDENCE_THRESHOLD = 80.0
    # -------------------------------------------------------


    print("=" * 50)
    print("  PROJECT 4 - PATH 1: OCR PIPELINE")
    print("=" * 50)
 
    # Load image
    img = load_image(IMAGE_PATH)

    # Pre-process
    preprocessed = preprocess_image(img)
 
    # Save preprocessed for verification
    save_preprocessed_image(preprocessed, "preprocessed_output.jpg")

    print("\n[→] Running OCR...")
    extracted_text = run_ocr(preprocessed, psm_mode=PSM_MODE)

    # Get confidence score
    avg_confidence = get_confidence(preprocessed, psm_mode=PSM_MODE)

     # -------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"\n📄 Extracted Text:\n{'-'*30}\n{extracted_text}\n{'-'*30}")
    print(f"\n📊 Average Confidence Score: {avg_confidence:.2f}%")
 
    if avg_confidence >= CONFIDENCE_THRESHOLD:
        print(f"✅ PASS: Confidence {avg_confidence:.2f}% >= {CONFIDENCE_THRESHOLD}% threshold")
    else:
        print(f"❌ WARN: Confidence {avg_confidence:.2f}% < {CONFIDENCE_THRESHOLD}% threshold")
        print("   Try a clearer image or adjust PSM mode.")


    # Save result to text file
    with open("ocr_result.txt", "w") as f: # This line opens (or creates) a file named "ocr_result.txt" in overwrite mode ("w") and safely writes the extracted text into it, automatically closing the file when finished.
        f.write(f"Extracted Text:\n{extracted_text}\n\nConfidence: {avg_confidence:.2f}%") # This line formats the extracted text and the calculated average confidence score (rounded to two decimal places) into a readable string and writes it directly into the open text file.
    print("\n[✓] Result saved to: ocr_result.txt")


if __name__ == "__main__":
    main()
