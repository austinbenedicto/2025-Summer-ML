import numpy as np
import pandas as pd
import struct
import cv2

# File paths
IMAGE_FILE = 'archive/train-images.idx3-ubyte'
LABEL_FILE = 'archive/train-labels.idx1-ubyte'

def load_images(image_path):
    """Load MNIST images from IDX file format"""
    with open(image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(label_path):
    """Load MNIST labels from IDX file format"""
    with open(label_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def basic_thinning(binary):
    """Apply basic thinning/skeletonization to binary image"""
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

def estimate_writing_direction_fft(binary):
    """Estimate writing direction using FFT analysis - returns global and quarter-based angles"""
    skeleton = basic_thinning(binary).astype(np.uint8) * 255
    grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
    grad_complex = grad_x + 1j * grad_y
    
    def calculate_angle(grad_comp):
        """Helper function to calculate dominant angle from gradient"""
        if np.sum(np.abs(grad_comp)) == 0:
            return 0.0
        fft_result = np.fft.fftshift(np.fft.fft2(grad_comp))
        power_spectrum = np.abs(fft_result)**2
        h, w = power_spectrum.shape
        y, x = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing='ij')
        angles = np.arctan2(y, x)
        angles_deg = np.degrees(angles)
        hist, bins = np.histogram(angles_deg, bins=180, range=(-180, 180), weights=power_spectrum)
        dominant_bin = np.argmax(hist)
        dominant_angle = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
        return dominant_angle
    
    # Global angle
    global_angle = calculate_angle(grad_complex)
    
    # Quarter-based angles
    h, w = grad_complex.shape
    mid_h, mid_w = h // 2, w // 2
    
    quarter_angles = []
    # Top-left quarter
    quarter_angles.append(calculate_angle(grad_complex[:mid_h, :mid_w]))
    # Top-right quarter
    quarter_angles.append(calculate_angle(grad_complex[:mid_h, mid_w:]))
    # Bottom-left quarter
    quarter_angles.append(calculate_angle(grad_complex[mid_h:, :mid_w]))
    # Bottom-right quarter
    quarter_angles.append(calculate_angle(grad_complex[mid_h:, mid_w:]))
    
    return [global_angle] + quarter_angles

def estimate_writing_direction_magnitude(binary):
    """Estimate writing direction magnitude using gradient analysis - returns global and quarter-based magnitudes"""
    skeleton = basic_thinning(binary).astype(np.uint8) * 255
    grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    def calculate_magnitude(mag_array):
        """Helper function to calculate mean magnitude"""
        return np.mean(mag_array) if mag_array.size > 0 else 0.0
    
    # Global magnitude
    global_magnitude = calculate_magnitude(magnitude)
    
    # Quarter-based magnitudes
    h, w = magnitude.shape
    mid_h, mid_w = h // 2, w // 2
    
    quarter_magnitudes = []
    # Top-left quarter
    quarter_magnitudes.append(calculate_magnitude(magnitude[:mid_h, :mid_w]))
    # Top-right quarter
    quarter_magnitudes.append(calculate_magnitude(magnitude[:mid_h, mid_w:]))
    # Bottom-left quarter
    quarter_magnitudes.append(calculate_magnitude(magnitude[mid_h:, :mid_w]))
    # Bottom-right quarter
    quarter_magnitudes.append(calculate_magnitude(magnitude[mid_h:, mid_w:]))
    
    return [global_magnitude] + quarter_magnitudes

def flood_fill(binary, y, x, visited):
    """Flood-fill to label connected regions (used for loop counting)"""
    stack = [(y, x)]
    h, w = binary.shape

    while stack:
        cy, cx = stack.pop()
        if (cy < 0 or cy >= h or cx < 0 or cx >= w):
            continue
        if visited[cy, cx] or not binary[cy, cx]:
            continue

        visited[cy, cx] = True
        # 4-connectivity
        stack.extend([
            (cy - 1, cx), (cy + 1, cx),
            (cy, cx - 1), (cy, cx + 1)
        ])

def count_loops(binary_img):
    """Count number of white regions (loops) inside digit using flood fill"""
    inverted = ~binary_img.copy()
    h, w = inverted.shape
    visited = np.zeros_like(inverted, dtype=bool)

    loop_count = 0
    for y in range(h):
        for x in range(w):
            if inverted[y, x] and not visited[y, x]:
                flood_fill(inverted, y, x, visited)
                loop_count += 1

    return max(0, loop_count - 1)  # Subtract outer background

def count_corners(binary_img):
    """Estimate corner count by local 3x3 neighborhood changes"""
    count = 0
    for y in range(1, binary_img.shape[0] - 1):
        for x in range(1, binary_img.shape[1] - 1):
            if binary_img[y, x]:
                neighborhood = binary_img[y-1:y+2, x-1:x+2]
                changes = np.sum(neighborhood != neighborhood[1, 1])
                if changes >= 4:
                    count += 1
    return count

def symmetry_metric(binary_img):
    """Calculate horizontal symmetry: compare left and right halves"""
    mid = binary_img.shape[1] // 2
    left = binary_img[:, :mid]
    right = np.fliplr(binary_img[:, -mid:])
    diff = np.sum(left != right)
    norm = left.size
    return diff / norm

def extract_features(img, threshold=50):
    """
    Extract all features from one MNIST image.
    Returns a list of 19 features (original 9 + 10 writing direction features).
    """
    binary = img < threshold  # dark = True
    dark_pixel_coords = np.argwhere(binary)

    if dark_pixel_coords.size == 0:
        return [0, -1, -1, 0, 0, 0, 0, 0, 1.0] + [0.0] * 10

    # Feature 1: Dark pixel count
    dark_pixel_count = len(dark_pixel_coords)

    # Features 2–3: Average X and Y coordinates
    avg_x = np.mean(dark_pixel_coords[:, 1])
    avg_y = np.mean(dark_pixel_coords[:, 0])

    # Features 4–5: Bounding box dimensions
    min_y, min_x = np.min(dark_pixel_coords, axis=0)
    max_y, max_x = np.max(dark_pixel_coords, axis=0)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Feature 6: Horizontal line intersection count
    intersection_count = 0
    for row in binary:
        transitions = np.diff(row.astype(int))
        intersection_count += np.sum(np.abs(transitions)) // 2

    # Feature 7: Loop count (enclosed white regions)
    loop_count = count_loops(binary)

    # Feature 8: Corner count
    corner_count = count_corners(binary)

    # Feature 9: Symmetry
    symmetry = symmetry_metric(binary)

    # Features 10-14: Writing direction angles (global + 4 quarters)
    writing_angles = estimate_writing_direction_fft(binary)

    # Features 15-19: Writing direction magnitudes (global + 4 quarters)
    writing_magnitudes = estimate_writing_direction_magnitude(binary)

    return [
        dark_pixel_count, avg_x, avg_y, width, height,
        intersection_count, loop_count, corner_count, symmetry
    ] + writing_angles + writing_magnitudes

# Load dataset
images = load_images(IMAGE_FILE)
labels = load_labels(LABEL_FILE)

# Process all images
features = []
for img in images:
    features.append(extract_features(img))

# Feature names
columns = [
    'dark_pixel_count',
    'avg_x',
    'avg_y',
    'bbox_width',
    'bbox_height',
    'intersection_count',
    'loop_count',
    'corner_count',
    'symmetry_metric',
    'writing_angle_global',
    'writing_angle_tl',
    'writing_angle_tr',
    'writing_angle_bl',
    'writing_angle_br',
    'writing_magnitude_global',
    'writing_magnitude_tl',
    'writing_magnitude_tr',
    'writing_magnitude_bl',
    'writing_magnitude_br'
]

# Build DataFrame and save
df = pd.DataFrame(features, columns=columns)
df['label'] = labels
df.to_csv("mnist_extended_features_no_scipy.csv", index=False)
print("✅ All features extracted. Saved to 'mnist_extended_features_no_scipy.csv'")