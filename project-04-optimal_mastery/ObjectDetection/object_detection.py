"""
Project 4 - Path 2: Object Detection using MobileNet-SSD
DecodeLabs | Batch 2026
Tool: OpenCV DNN module + MobileNet-SSD (pre-trained model)
 
Files needed (download separately):
  - MobileNetSSD_deploy.prototxt  (model architecture)
  - MobileNetSSD_deploy.caffemodel (trained weights)
  Download: https://github.com/chuanqi305/MobileNet-SSD
"""

import cv2
import numpy as np
import os

# -------------------------------------------------------
# CLASS LABELS: 21 classes MobileNet-SSD can detect
# -------------------------------------------------------
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Random colors for each class label (for bounding box display)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# size=(len(CLASSES), 3) : Defines the shape of the matrix (array) that NumPy will generate.
# It requests a 2D array with 21 rows (one for each class) and 3 columns (representing the Red, Green, and Blue channels for each color).
# What it does: Fills that (21 x 3) matrix with random floating-point numbers picked evenly (uniformly) between 0 and 255.In digital image processing, any color can be represented by combining Red, Green, and Blue values ranging from 0 to 255.


# -------------------------------------------------------
# CONFIDENCE THRESHOLD (80% as per project requirement)
# -------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80

def load_model(prototxt_path, model_path):
    """
    Load the pre-trained MobileNet-SSD model using OpenCV DNN.
    """
    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Prototxt not found: {prototxt_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
 
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # cv2.dnn: This is OpenCV's Deep Neural Network module. It contains built-in tools to run pre-trained AI models directly inside your computer vision code.
    # .readNetFromCaffe(...): This is the specific function that loads models trained using Caffe, a popular deep learning framework. It requires two distinct files to work.
    # prototxt_path: This is the file path to a text file (.prototxt). It acts as the blueprint or architecture of the network. It describes how many layers the model has, how they connect, and what type of neural network it is (like a CNN)
    # model_path: This is the file path to a binary file (usually ending in .caffemodel). It contains the actual brains or learned weights of the model. This file holds the mathematical patterns the AI learned during its training phase
    print("[✓] MobileNet-SSD model loaded successfully")
    return net


# A Deep Neural Network (DNN) is a general-purpose AI brain with multiple layers, used for analyzing data like text or numbers. 
# A Convolutional Neural Network (CNN) is a specialized DNN designed specifically for visual data, acting like a computer's visual cortex to scan and recognize patterns in images .

def load_image(image_path):
    """Load image from disk."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"[✓] Image loaded: {image_path} | Shape: {img.shape}")
    return img

def preprocess_to_blob(img):
    """
    Convert image to a 4D Blob for the neural network.
    - Resizes to 300x300 (MobileNet-SSD input requirement)
    - Subtracts mean pixel values (normalization)
    - Constructs shape: (1, 3, 300, 300)
    """
    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),   # resize to 300x300
        0.007843,                       # scale factor
        (300, 300),                     # target size
        127.5                           # mean subtraction value
    )
    print("[✓] Pre-processing: 4D Blob constructed (1, 3, 300, 300)")
    return blob

# This block of code finishes preparing your raw image so that the neural network can actually read it. Neural networks cannot accept raw, random-sized photos; they demand highly structured data. This process is called pre-processing, and it outputs a special 4D matrix called a blob.
#  It shrinks or stretches your input image to exactly 300 pixels wide by 300 pixels high.Why: AI models have fixed input dimensions. If your Caffe model was trained to look at (300 x 300) images, passing any other size will cause the script to crash.
# It subtracts 127.5 from every single pixel color value (R, G, and B) in the image. Since pixel values range from 0 to 255, subtracting 127.5 shifts the numbers so they range from -127.5 to +127.5.Why: This centers the data around zero. It cancels out drastic lighting or exposure differences, helping the AI focus on shapes and features rather than how bright the photo is.
# after multiplying with the scale factor -  the final pixel numbers end up neatly normalized between -1.0 and 1.0. Neural networks train and perform best when numbers stay within this small, standard range.
# cv2.dnn.blobFromImage - Besides the math above, it rearranges the internal structure of the image array and adds a fourth dimension.

# The Output (1, 3, 300, 300) breakdown:
# 1: Batch size (you are feeding exactly 1 image to the network).
# 3: Channels (the Red, Green, and Blue color layers). OpenCV switches them from BGR to RGB here to match Caffe's expectations.
# 300, 300: The height and width of the image.

def run_detection(net, blob):
    """
    Feed blob into the network and get detections.
    This is the Single Shot Detection (one forward pass only).
    """
    net.setInput(blob)
    detections = net.forward()
    print(f"[✓] Forward pass complete. Total detections: {detections.shape[2]}")
    return detections

# detections = net.forward(): This triggers the actual mathematical computation. The data flows forward through all the convolutional and hidden layers of the network. The final output is saved into a variable named detections.
# net.setInput(blob): This places your pre-processed 4D blob into the very first layer (the input layer) of the neural network. You are loading the factory conveyor belt with your image data.
# detections.shape[2]: The output of a Caffe SSD model is a 4D array with a shape like (1, 1, N, 7). The third number (shape[2]) represents \(N\), which is the maximum number of potential objects the network detected in the image (often 100 or 200 raw proposals).

# To understand what it returns, think of it as a spreadsheet with \(N\) rows (where \(N\) is the total number of objects the AI thinks it found, usually 100 or 200) and exactly 7 columns of data for each object.
def draw_bounding_boxes(img, detections):
    """
    Filter detections by confidence threshold (>= 80%)
    and draw bounding boxes with labels on the image.
    """
    
    h, w = img.shape[:2]
    output_img = img.copy()
    valid_detections = 0
    
    # This block of code extracts the physical dimensions of your original image(height and width in pixels), creates a safe backup copy to draw on, and initializes a counter to track how many real objects your AI finds.
    
    print(f"\n{'='*50}")
    print("  DETECTIONS (confidence >= 80%)")
    print(f"{'='*50}")
    
    
    for i in range(detections.shape[2]): # for i in range(detections.shape[2]): Loops through every single raw prediction proposal (usually 100 or 200 items) returned by the neural network.
        confidence = float(detections[0, 0, i, 2]) # confidence = float(detections[0, 0, i, 2]) Extracts the accuracy/certainty score for the current object proposal and converts it into a decimal number.
 
        # Apply the 80% confidence threshold filter
        if confidence >= CONFIDENCE_THRESHOLD:
            valid_detections += 1
            class_idx = int(detections[0, 0, i, 1]) # Extracts the numerical category ID of the found object and converts it into a clean whole integer.
            label = CLASSES[class_idx] # Looks up the actual text name (like "car" or "person") from your predefined list of labels using the numerical ID.
 
            # Convert normalized coordinates to actual pixel values
            # Model outputs values between 0 and 1
            # Multiply by actual image dimensions to get pixel positions
            
            x1 = int(detections[0, 0, i, 3] * w) # Converts the top-left X decimal coordinate into an exact pixel location by multiplying it against the original image width (w)
            y1 = int(detections[0, 0, i, 4] * h) # Converts the top-left Y decimal coordinate into an exact pixel location by multiplying it against the original image height (h).
            x2 = int(detections[0, 0, i, 5] * w) # Converts the bottom-right X decimal coordinate into an exact pixel location using the image width (w).
            y2 = int(detections[0, 0, i, 6] * h) # Converts the bottom-right Y decimal coordinate into an exact pixel location using the image height (h).
 
            # Draw bounding box
            color = COLORS[class_idx] # Selects a specific color assigned to this object type from your preset color tracking array.
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2) #  Draws a 2-pixel-thick bounding box outline directly over the object on your copied image.
 
            # Draw label with confidence
            label_text = f"{label}: {confidence*100:.1f}%" #  Formats a clean text string showing the object's text name and its confidence percentage rounded to one decimal place (e.g., "dog: 92.4%").
            cv2.putText(
                output_img, label_text,
                (x1, y1 - 10), # (x1, y1 - 10): The precise X, Y pixel coordinates where the bottom-left corner of the text string starts. Subtracting 10 from y1 moves the text exactly 10 pixels upward, ensuring it hovers cleanly right above the top line of your bounding box.
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            # Renders the formatted text label exactly 10 pixels above the top-left corner of the drawn bounding box using a standard font.
 
            print(f"  ✅ {label:<15} | Confidence: {confidence*100:.1f}% | Box: ({x1},{y1}) to ({x2},{y2})")
 
    if valid_detections == 0:
        print("  ⚠️  No detections passed the 80% threshold.")
        print("  Try with a clearer image or lower threshold for testing.")
 
    print(f"\n[✓] Total valid detections: {valid_detections}")
    return output_img, valid_detections



def main():
    # -------------------------------------------------------
    # INPUTS — change these paths
    IMAGE_PATH   = "sample_dog.jpeg"
    PROTOTXT     = "MobileNetSSD_deploy.prototxt"
    MODEL_WEIGHTS = "mobilenet_iter_73000.caffemodel"
    OUTPUT_PATH  = "detection_output.jpg"
    # -------------------------------------------------------
 
    print("=" * 50)
    print("  PROJECT 4 - PATH 2: OBJECT DETECTION PIPELINE")
    print("=" * 50)
 
    # Load model
    net = load_model(PROTOTXT, MODEL_WEIGHTS)
 
    # Load image
    img = load_image(IMAGE_PATH)
 
    # Pre-process to blob
    blob = preprocess_to_blob(img)
 
    # Run detection (Single Shot)
    detections = run_detection(net, blob)
 
    # Draw bounding boxes (only those >= 80% confidence)
    output_img, count = draw_bounding_boxes(img, detections)
 
    # Save output image
    cv2.imwrite(OUTPUT_PATH, output_img)
    print(f"\n[✓] Output image saved: {OUTPUT_PATH}")
 
    print("\n" + "=" * 50)
    print(f"  FINAL RESULT: {count} object(s) detected with >= 80% confidence")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()


