import os
import random
import cv2
import matplotlib.pyplot as plt

# Point to your dataset
img_dir = "Shelf-Retail-Prototype-2/train/images"
label_dir = "Shelf-Retail-Prototype-2/train/labels"

# Get all image filenames
all_images = os.listdir(img_dir)
print(f"Total training images: {len(all_images)}")

# Pick 6 random images to look at
sample = random.sample(all_images, 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Sample shelf images from your dataset", fontsize=14)

for ax, img_name in zip(axes.flat, sample):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(img_name[:30], fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_images.png")
print("Saved sample_images.png - open it to see your images")

# Count labels too
all_labels = os.listdir(label_dir)
print(f"Total label files: {len(all_labels)}")

# Peek inside one label file
sample_label = all_labels[0]
label_path = os.path.join(label_dir, sample_label)
print(f"\nSample label file: {sample_label}")
print("Contents:")
with open(label_path, "r") as f:
    print(f.read())
print("Each line = one detected object: class x y width height")