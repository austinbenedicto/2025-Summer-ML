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

def estimate_writing_direction_fft(binary):
    skeleton = basic_thinning(binary).astype(np.uint8) * 255
    grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
    grad_complex = grad_x + 1j * grad_y
    fft_result = np.fft.fftshift(np.fft.fft2(grad_complex))
    power_spectrum = np.abs(fft_result)**2
    h, w = power_spectrum.shape
    y, x = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing='ij')
    angles = np.arctan2(y, x)
    angles_deg = np.degrees(angles)
    hist, bins = np.histogram(angles_deg, bins=180, range=(-180, 180), weights=power_spectrum)
    dominant_bin = np.argmax(hist)
    dominant_angle = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
    return dominant_angle

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

def zone_density_features(binary, grid_size=(4, 4)):
    h, w = binary.shape
    gh, gw = grid_size
    zone_h, zone_w = h // gh, w // gw
    features = []
    for i in range(gh):
        for j in range(gw):
            zone = binary[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
            density = np.sum(zone) / zone.size
            features.append(density)
    return features

def hu_moments_features(binary):
    binary_uint8 = binary.astype(np.uint8) * 255
    moments = cv2.moments(binary_uint8)
    hu = cv2.HuMoments(moments).flatten()
    return hu.tolist()

def extract_features(img, threshold=200):
    binary = img > threshold
    coords = np.argwhere(binary)
    if coords.size == 0:
        # Return 12 values to match the normal case
        return [0, -1, -1, 0, 0, 0, 0, 0, 0.0, 0.0, [0]*7, [0]*16]
    
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
    writing_angle = estimate_writing_direction_fft(binary)
    hu_moments_feat = hu_moments_features(binary)  # Renamed to avoid conflict
    zone_density = zone_density_features(binary)
    
    return [
        dark_pixel_count, avg_x, avg_y, width, height,
        intersection_count, loop_count, len(corner_count), symmetry, writing_angle, 
        hu_moments_feat, zone_density
    ]

def visualize_features(img, features, corners, binary):
    dark_pixel_count, avg_x, avg_y, width, height, _, _, _, _, writing_angle, hu_moments_feat, zone_density = features

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Feature Visualization\nWriting Direction: {writing_angle:.1f}°")

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

    skeleton = basic_thinning(binary).astype(np.uint8) * 255
    grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
    step = 3
    y_grid, x_grid = np.mgrid[0:skeleton.shape[0]:step, 0:skeleton.shape[1]:step]
    u = grad_x[::step, ::step]
    v = grad_y[::step, ::step]
    ax.quiver(x_grid, y_grid, u, -v, color='red', angles='xy', scale_units='xy', scale=1, alpha=0.5)

    ax.legend()
    plt.show()

columns = [
    'dark_pixel_count', 'avg_x', 'avg_y', 'bbox_width', 'bbox_height',
    'intersection_count', 'loop_count', 'corner_count', 'symmetry_metric', 'writing_angle',
    'hu_moments_features', 'zone_density'
]

def test_image(index, images, labels):
    img = images[index]
    label = labels[index]
    binary = img > 200
    feats = extract_features(img)
    corners, _ = detect_corners_filtered(binary)
    visualize_features(img, feats, corners, binary)
    print(f"--- Features for digit '{label}' at index {index} ---")
    for name, val in zip(columns, feats):
        if isinstance(val, float):
            print(f"{name:20s}: {val:.3f}")
        elif isinstance(val, list) or isinstance(val, np.ndarray):
            print(f"{name:20s}: {np.array2string(np.array(val), precision=3)}")
        else:
            print(f"{name:20s}: {val}")

if __name__ == "__main__":
    images = load_images(IMAGE_FILE)
    labels = load_labels(LABEL_FILE)
    for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        test_image(idx, images, labels)
