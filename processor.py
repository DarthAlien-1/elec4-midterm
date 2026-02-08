import cv2
import os
import glob
import numpy as np

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

def apply_thermal_effect(image):
    thermal = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return thermal

def apply_motion_blur(image, kernel_size=15):
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur /= kernel_size
    
    blur = cv2.filter2D(image, -1, kernel_motion_blur)
    return blur

def process_images():
    image_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    print(f"Found {len(image_files)} images...")

    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is None:
            continue
        
        filename = os.path.basename(file_path)

        # --- TECHNIQUE 1: Thermal Vision ---
        thermal_img = apply_thermal_effect(img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"thermal_{filename}"), thermal_img)

        # --- TECHNIQUE 2: Motion Blur ---
        motion_img = apply_motion_blur(img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"motion_{filename}"), motion_img)

        print(f"Processed: {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    process_images()
