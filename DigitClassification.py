import numpy as np
import pandas as pd
import struct
import os

# Paths to your MNIST files (update if needed)
IMAGE_FILE = 'train-images.idx3-ubyte'
LABEL_FILE = 'train-labels.idx1-ubyte'

def load_images(image_path):
    with open(image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read the rest of the data and reshape
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(label_path):
    with open(label_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def extract_features(img, threshold=50):
    binary = img < threshold  # dark = True
    dark_pixel_coords = np.argwhere(binary)

    if dark_pixel_coords.size == 0:
        return [0, -1, -1, 0, 0, 0]

    dark_pixel_count = len(dark_pixel_coords)
    avg_x = np.mean(dark_pixel_coords[:, 1])  # columns
    avg_y = np.mean(dark_pixel_coords[:, 0])  # rows

    min_y, min_x = np.min(dark_pixel_coords, axis=0)
    max_y, max_x = np.max(dark_pixel_coords, axis=0)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    intersection_count = 0
    for row in binary:
        transitions = np.diff(row.astype(int))
        intersection_count += np.sum(np.abs(transitions)) // 2

    return [dark_pixel_count, avg_x, avg_y, width, height, intersection_count]

# Load data
images = load_images(IMAGE_FILE)
labels = load_labels(LABEL_FILE)

features = []
for img in images:
    features.append(extract_features(img))

# Save to DataFrame
columns = ['dark_pixel_count', 'avg_x', 'avg_y', 'bbox_width', 'bbox_height', 'intersection_count']
df = pd.DataFrame(features, columns=columns)
df['label'] = labels

# Export to CSV
df.to_csv("mnist_features_from_idx.csv", index=False)
print("✅ Done! Features saved to 'mnist_features_from_idx.csv'")