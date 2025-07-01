

import numpy as np
import cv2
import struct
import matplotlib.pyplot as plt

# File paths for MNIST data
IMAGE_FILE = 'archive/train-images.idx3-ubyte'
LABEL_FILE = 'archive/train-labels.idx1-ubyte'

def load_images(image_path):
    with open(image_path, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(label_path):
    with open(label_path, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def basic_thinning(binary):
    img = binary.astype(np.uint8) * 255
    prev = np.zeros_like(img)
    kernel = np.ones((3, 3), np.uint8)
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(prev, temp)
        if cv2.countNonZero(cv2.absdiff(img, eroded)) == 0:
            break
        img = eroded.copy()
        prev = skel.copy()
    return skel > 0

def detect_corners_filtered(binary, min_distance=3):
    skeleton = basic_thinning(binary)
    raw_corners = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x]:
                nhood = skeleton[y-1:y+2, x-1:x+2]
                if (nhood[0, 1] and nhood[1, 2]) or (nhood[1, 0] and nhood[0, 1]) or \
                   (nhood[1, 2] and nhood[2, 1]) or (nhood[2, 1] and nhood[1, 0]):
                    raw_corners.append((x, y))
    filtered = []
    for (x, y) in raw_corners:
        if all((x - fx)**2 + (y - fy)**2 >= min_distance**2 for (fx, fy) in filtered):
            filtered.append((x, y))
    return filtered, skeleton

def flood_fill(binary, y, x, visited):
    stack = [(y, x)]
    h, w = binary.shape
    while stack:
        cy, cx = stack.pop()
        if not (0 <= cy < h and 0 <= cx < w): continue
        if visited[cy, cx] or not binary[cy, cx]: continue
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

def count_intersections(binary):
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

def symmetry_metric(binary):
    mid = binary.shape[1] // 2
    left = binary[:, :mid]
    right = np.fliplr(binary[:, -mid:])
    similarity = np.sum(left == right)
    total = left.size
    return (similarity / total) * 100

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
    intersection_count = count_intersections(binary)
    loop_count = count_loops(binary)
    corner_count, _ = detect_corners_filtered(binary)
    symmetry = symmetry_metric(binary)
    return [
        dark_pixel_count, avg_x, avg_y, width, height,
        intersection_count, loop_count, len(corner_count), symmetry
    ]

def visualize_features(img, features, corners, binary):
    dark_pixel_count, avg_x, avg_y, width, height, _, _, _, _ = features

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title("Feature Visualization")

    ax.plot(avg_x, avg_y, 'go', label='Center of Mass')

    min_y, min_x = np.argwhere(binary).min(axis=0)
    max_y, max_x = np.argwhere(binary).max(axis=0)
    rect = plt.Rectangle((min_x, min_y), max_x - min_x + 1, max_y - min_y + 1,
                         linewidth=1.5, edgecolor='yellow', facecolor='none', label='Bounding Box')
    ax.add_patch(rect)

    for (x, y) in corners:
        ax.plot(x, y, 'ro', markersize=3)

    for y, row in enumerate(binary):
        inside = False
        for pixel in row:
            if pixel and not inside:
                inside = True
                ax.axhline(y=y, color='cyan', linestyle='--', alpha=0.3)
                break
            elif not pixel:
                inside = False

    ax.legend()
    plt.show()



# ===== Test Image Function (like DigitTester) =====
columns = [
    'dark_pixel_count', 'avg_x', 'avg_y', 'bbox_width', 'bbox_height',
    'intersection_count', 'loop_count', 'corner_count', 'symmetry_metric'
]

def test_image(index, images, labels):
    img = images[index]
    label = labels[index]
    binary = img > 200
    coords = np.argwhere(binary)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    avg_x = np.mean(coords[:, 1])
    avg_y = np.mean(coords[:, 0])

    # Visualization
    feats = extract_features(img)
    corners, _ = detect_corners_filtered(binary)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x + 1, max_y - min_y + 1,
                               linewidth=2, edgecolor='r', facecolor='none'))
    ax.plot(avg_x, avg_y, 'bo', label='Center')
    # Plot corners
    for (x, y) in corners:
        ax.plot(x, y, 'ro', markersize=4, label='Corner' if (x, y) == corners[0] else "")
    # Plot intersections (horizontal lines)
    for y_idx, row in enumerate(binary):
        inside = False
        for pixel in row:
            if pixel and not inside:
                inside = True
                ax.axhline(y=y_idx, color='cyan', linestyle='--', alpha=0.3, label='Intersection' if y_idx == 0 else "")
                break
            elif not pixel:
                inside = False
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.title(f"Digit: {label} | Index: {index}")
    plt.show()

    # Feature Output
    print(f"--- Features for digit '{label}' at index {index} ---")
    for name, val in zip(columns, feats):
        print(f"{name:20s}: {val:.3f}" if isinstance(val, float) else f"{name:20s}: {val}")


# ===== Example Usage =====
if __name__ == "__main__":
    images = load_images(IMAGE_FILE)
    labels = load_labels(LABEL_FILE)

    # Test specific images (edit this list as needed)
    for idx in [0, 1, 2,3,4,5,6,7,8,9]:
        test_image(idx, images, labels)
