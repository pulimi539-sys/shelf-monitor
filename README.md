# Retail Shelf Monitor

Detects out-of-stock and low-stock zones on retail shelf 
images using a fine-tuned YOLOv8 model.

## Results
- mAP@50: 90.9%
- Classes: Apple, Choco Fills, Choco Pie
- Training images: 1,128
- Inference speed: 2.6ms per image

## Tech Stack
YOLOv8 · OpenCV · Gradio · SQLite · Python 3.10

## How to run
pip install -r requirement.txt
python app.py
