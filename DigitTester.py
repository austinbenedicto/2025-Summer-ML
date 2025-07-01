# Re-run full setup after environment reset, using improved feature logic

import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
IMAGE_FILE = 'archive/train-images.idx3-ubyte'
LABEL_FILE = 'archive/train-labels.idx1-ubyte'

# Loaders
def load_images(image_path):
    with open(image_path, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(label_path):
    with open(label_path, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# Flood fill and loop counter
def flood_fill(binary, y, x, visited):
    stack = [(y, x)]
    h, w = binary.shape
    while stack:
        cy, cx = stack.pop()
        if not (0 <= cy < h and 0 <= cx < w):
            continue
        if visited[cy, cx] or not binary[cy, cx]:
            continue
        visited[cy, cx] = True
        stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])

def count_loops(binary_img):
    inverted = ~binary_img
    visited = np.zeros_like(inverted, dtype=bool)
    loops = 0
    for y in range(inverted.shape[0]):
        for x in range(inverted.shape[1]):
            if inverted[y, x] and not visited[y, x]:
                flood_fill(inverted, y, x, visited)
                loops += 1
    return max(0, loops - 1)

# Improved corner count
def count_corners_improved(binary):
    """
    Count corners using a simple neighborhood-based approach.
    This is more reliable than the OpenCV-based approach.
    """
    count = 0
    for y in range(1, binary.shape[0] - 1):
        for x in range(1, binary.shape[1] - 1):
            if binary[y, x]:
                # Check 3x3 neighborhood
                nhood = binary[y-1:y+2, x-1:x+2]
                center = nhood[1, 1]
                
                # Count transitions around the center pixel
                neighbors = [
                    nhood[0, 1], nhood[0, 2], nhood[1, 2], nhood[2, 2],
                    nhood[2, 1], nhood[2, 0], nhood[1, 0], nhood[0, 0]
                ]
                
                # Count transitions from 0 to 1
                transitions = 0
                for i in range(8):
                    if neighbors[i] != neighbors[(i + 1) % 8]:
                        transitions += 1
                
                # A corner typically has 2 or 4 transitions
                if transitions == 2 or transitions == 4:
                    # Additional check: make sure we have some neighbors
                    if sum(neighbors) >= 2:
                        count += 1
    
    return count

# Improved intersection count
def count_intersections_improved(binary):
    intersection_count = 0
    for row in binary:
        inside_segment = False
        segments = 0
        for pixel in row:
            if pixel and not inside_segment:
                inside_segment = True
                segments += 1
            elif not pixel:
                inside_segment = False
        intersection_count += segments
    return intersection_count

# Improved symmetry
def symmetry_metric_improved(binary):
    mid = binary.shape[1] // 2
    left = binary[:, :mid]
    right = np.fliplr(binary[:, -mid:])
    similarity = np.sum(left == right)
    total = left.size
    return (similarity / total) * 100

# Feature extraction using improved metrics
def extract_features(img, threshold=200):
    binary = img > threshold
    coords = np.argwhere(binary)
    if coords.size == 0:
        return [0, -1, -1, 0, 0, 0, 0, 0, 0.0]
    dark_pixel_count = len(coords)
    avg_x = np.mean(coords[:, 1])
    avg_y = np.mean(coords[:, 0])
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    intersection_count = count_intersections_improved(binary)
    loop_count = count_loops(binary)
    corner_count = count_corners_improved(binary)
    symmetry = symmetry_metric_improved(binary)
    return [dark_pixel_count, avg_x, avg_y, width, height,
            intersection_count, loop_count, corner_count, symmetry]

# Column labels with units
columns = [
    'dark_pixel_count (px)',
    'avg_x (px)',
    'avg_y (px)',
    'bbox_width (px)',
    'bbox_height (px)',
    'intersection_count (segments)',
    'loop_count (holes)',
    'corner_count (L-shapes)',
    'symmetry_score (%)'
]

# Test function
def test_image(index, images, labels):
    img = images[index]
    label = labels[index]
    binary = img > 200
    coords = np.argwhere(binary)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    avg_x = np.mean(coords[:, 1])
    avg_y = np.mean(coords[:, 0])

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.add_patch(patches.Rectangle((min_x, min_y), max_x - min_x + 1, max_y - min_y + 1,
                                   linewidth=2, edgecolor='r', facecolor='none'))
    ax.plot(avg_x, avg_y, 'bo')
    plt.title(f"Digit: {label} | Index: {index}")
    plt.show()

    feats = extract_features(img)
    print(f"--- Features for digit '{label}' at index {index} ---")
    for name, val in zip(columns, feats):
        print(f"{name:35s}: {val:.3f}" if isinstance(val, float) else f"{name:35s}: {val}")

# Load files
images = load_images(IMAGE_FILE)
labels = load_labels(LABEL_FILE)

# Test a couple of images
test_image(0, images, labels)
test_image(14, images, labels)
test_image(28, images, labels)
test_image(42, images, labels)
test_image(56, images, labels)