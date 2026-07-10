## Training Results (Day 4)
- Model: YOLOv8n (nano) — chose nano for speed vs accuracy tradeoff
- Dataset: 1,128 training images, 3 classes
- Epochs: 30
- Final mAP@50: 90.9%
- Apple: 90.5%, Choco Fills: 90.8%, Choco Pie: 91.4%
- Training time: 10 minutes on T4 GPU
- Inference speed: 2.6ms per image

## Why YOLOv8n not YOLOv8s?
Started with nano to prove the pipeline works fast.
90.9% mAP shows nano is sufficient for this dataset.
Would try small (yolov8s) in v2 to compare accuracy vs speed.