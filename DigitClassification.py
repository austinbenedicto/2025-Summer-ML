import os
import numpy as np
from PIL import Image
import pandas as pd

# Path to the dataset directory. This should be the unzipped "mnist_png/train" directory.
# Inside, it is expected to have subfolders named "0", "1", ..., "9", each containing PNG images of that digit.
DATASET_DIR = 'mnist_png/train'  # Change this if using the test set or another structure

def extract_features(image_path, threshold=50):
    """
    Extracts features from a single MNIST image.
    
    Parameters:
        image_path (str): Path to the image file.
        threshold (int): Intensity threshold to consider a pixel "dark".
    
    Returns:
        List containing:
            - dark pixel count
            - average X coordinate of dark pixels
            - average Y coordinate of dark pixels
            - bounding box width
            - bounding box height
            - intersection count (horizontal)
    """
    # Open image and convert to grayscale (0 = black, 255 = white)
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    # Create binary image: True where pixel is dark (value < threshold)
    binary = arr < threshold
    # Get coordinates (y, x) of all dark pixels
    dark_pixel_coords = np.argwhere(binary)

    if dark_pixel_coords.size == 0:
        # If no dark pixels, return default values
        return [0, -1, -1, 0, 0, 0]

    # 1. Dark pixel count
    dark_pixel_count = len(dark_pixel_coords)

    # 2. Average X and 3. Average Y of dark pixels
    avg_x = np.mean(dark_pixel_coords[:, 1])  # X is the column
    avg_y = np.mean(dark_pixel_coords[:, 0])  # Y is the row

    # 4. Bounding box width and 5. height
    min_y, min_x = np.min(dark_pixel_coords, axis=0)
    max_y, max_x = np.max(dark_pixel_coords, axis=0)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 6. Intersection count: horizontal line intersections
    intersection_count = 0
    for row in binary:
        # Count transitions between white (False) and black (True)
        transitions = np.diff(row.astype(int))
        # Each pair of transitions (+1 or -1) indicates one intersection
        intersection_count += np.sum(np.abs(transitions)) // 2

    return [dark_pixel_count, avg_x, avg_y, width, height, intersection_count]

# Store extracted features and corresponding labels
features = []
labels = []

# Loop through each digit folder (0–9)
for label in sorted(os.listdir(DATASET_DIR)):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue  # Skip if it's not a directory

    # Loop through each image in that digit folder
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        # Extract features from the image
        feats = extract_features(image_path)
        features.append(feats)
        labels.append(int(label))  # Store the label as an integer

# Convert collected data to a Pandas DataFrame
columns = [
    'dark_pixel_count',
    'avg_x',
    'avg_y',
    'bbox_width',
    'bbox_height',
    'intersection_count'
]
df = pd.DataFrame(features, columns=columns)
df['label'] = labels  # Add label column to end

# Save results to a CSV file for further analysis or ML training
df.to_csv("mnist_features.csv", index=False)
print("✅ Feature extraction complete. Saved to 'mnist_features.csv'")