# Automated Image Processing

## Project Overview
Automated Image Processing is a Python-based image processing application that demonstrates the integration of computer vision techniques with DevOps automation. The system processes images using multiple image processing operations and ensures reliability through automated testing and Continuous Integration (CI) using GitHub Actions.

---

## Learning Objectives
- Apply image processing techniques using Python and OpenCV  
- Implement automated testing using PyTest  
- Configure and utilize a Continuous Integration (CI) pipeline with GitHub Actions  
- Understand DevOps workflows and automation concepts  

---

## Tools and Technologies
- Python 3  
- OpenCV (opencv-python)  
- NumPy  
- PyTest  
- Git and GitHub  
- GitHub Actions  

## Project Structure 
```text
elec4-midterm/
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions pipeline and Project Dependencies
│
├── input_images/                 # Drop images here (trigger CI)
│   └── sample.jpg
│
├── output_images/                # Processed images appear here
│   └── sample_thermal.jpg       
│
├── processor.py                  # Image processing logic
├── test_process.py               # Pytest validations
├── requirements.txt              # Project Dependencies
│       
└── README.md

```
---

## Implemented Image Processing Features

### Thermal Vision (Heatmap)
Applies a JET colormap to visualize intensity variations within the image.

### Motion Blur (Directional)
Creates a horizontal motion blur effect using a custom kernel for dynamic visuals.

### Mirror Flip (Horizontal)
Flips images horizontally for mirror reflection effects.

### Channel Swap (BGR → RGB)
Converts OpenCV's BGR color space to standard RGB color format.

### Solarize
Inverts pixels with brightness above 128 for a solarization effect.

### Rotate 90° Clockwise
Rotates images 90 degrees in the clockwise direction.

---

## How to Use the Automated Image Processing System

### 1. Installation
```bash
git clone https://github.com/DarthAlien-1/elec4-midterm.git
cd elec4-midterm
pip install -r requirements.txt
```

### 2. Upload/Input Images
- Place the image or images you want to process inside the `input_images` folder.  
- Supported image formats include `.jpg`, `.png`, and `.jpeg`.

### 3. Program Start
- Run the processor script:
```bash
python processor.py
```
- The system automatically processes all images found in the `input_images` folder.

### 4. Automated Processing
Each input image undergoes the following operations:
- Thermal Vision (Heatmap)  
- Motion Blur (Directional)  
- Mirror Flip (Horizontal)  
- Channel Swap (BGR → RGB)  
- Solarize  
- Rotate 90° Clockwise

### 5. Output Results
- All processed images are automatically saved in the `output_images` folder.  
- Each image processing feature generates its own output file with a descriptive prefix for easy comparison and validation.

### 6. Testing
- Run automated tests using PyTest:
```bash
pytest test_process.py
```

---

## Author
DarthAlien-1