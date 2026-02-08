import cv2
import os
import glob

# Define our folders
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

def process_images():
    # Get all images (jpg or png)
    image_files = glob.glob(os.path.join(INPUT_DIR, "*"))

    print(f"Found {len(image_files)} images to process...")

    for file_path in image_files:
        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            continue

        filename = os.path.basename(file_path)

        # TECHNIQUE 1: Grayscale (Black & White)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"gray_{filename}"), gray_img)

        # TECHNIQUE 2: Invert (Negative)
        invert_img = cv2.bitwise_not(img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"invert_{filename}"), invert_img)

        print(f"Processed: {filename}")

if __name__ == "__main__":
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    process_images()