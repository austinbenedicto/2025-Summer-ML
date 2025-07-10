import numpy as np
import cv2
import struct
import pandas as pd

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

def estimate_writing_directions_fft(binary):
    """
    Returns global and quadrant-based dominant direction and magnitude.
    Each returned value is a tuple: (angle in degrees, magnitude).
    """
    def compute_direction_magnitude(region):
        skeleton = basic_thinning(region).astype(np.uint8) * 255
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
        magnitude = hist[dominant_bin]
        return (dominant_angle, magnitude)

    global_dir_mag = compute_direction_magnitude(binary)

    h, w = binary.shape
    quarters = [
        binary[:h//2, :w//2], binary[:h//2, w//2:],
        binary[h//2:, :w//2], binary[h//2:, w//2:]
    ]
    quadrant_dirs_mags = [compute_direction_magnitude(q) for q in quarters]

    return global_dir_mag, quadrant_dirs_mags

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
        # Return the correct number of zeros for all features
        return [0, -1, -1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0] + [0.0]*8 + [0.0]*8 + [0.0]*7
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
    (global_angle, global_magnitude), quadrants = estimate_writing_directions_fft(binary)
    quadrant_angles = [q[0] for q in quadrants]
    quadrant_magnitudes = [q[1] for q in quadrants]
    zone_densities = zone_density_features(binary)
    hu = hu_moments_features(binary)
    return [
        dark_pixel_count, avg_x, avg_y, width, height,
        intersection_count, loop_count, len(corner_count), symmetry,
        global_angle, global_magnitude
    ] + quadrant_angles + quadrant_magnitudes + zone_densities + hu


# Column names for CSV output
columns = [
    'dark_pixel_count', 'avg_x', 'avg_y', 'bbox_width', 'bbox_height',
    'intersection_count', 'loop_count', 'corner_count', 'symmetry_metric',
    'writing_angle', 'writing_magnitude',
    'q1_angle', 'q2_angle', 'q3_angle', 'q4_angle',
    'q1_magnitude', 'q2_magnitude', 'q3_magnitude', 'q4_magnitude',
    'zone_density_1', 'zone_density_2', 'zone_density_3', 'zone_density_4',
    'zone_density_5', 'zone_density_6', 'zone_density_7', 'zone_density_8',
    'zone_density_9', 'zone_density_10', 'zone_density_11', 'zone_density_12',
    'zone_density_13', 'zone_density_14', 'zone_density_15', 'zone_density_16',
    'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7'
]

if __name__ == "__main__":
    # Configuration: Set the percentage of images to omit (0-100)
    # For example, omit_percentage = 20 means use only 80% of images
    omit_percentage = 0  # Change this value to omit images (e.g., 20 for 20%)
    
    print("Loading MNIST data...")
    images = load_images(IMAGE_FILE)
    labels = load_labels(LABEL_FILE)
    
    # If omitting images, select equal numbers of each digit
    if omit_percentage > 0:
        print(f"Selecting {100 - omit_percentage}% of images with equal digit representation...")
        
        # Group images by digit
        digit_indices = {digit: [] for digit in range(10)}
        for i, label in enumerate(labels):
            digit_indices[label].append(i)
        
        # Calculate how many images to keep per digit
        min_count = min(len(indices) for indices in digit_indices.values())
        keep_per_digit = int(min_count * (100 - omit_percentage) / 100)
        
        print(f"Original: ~{min_count} images per digit")
        print(f"Keeping: {keep_per_digit} images per digit")
        
        # Select equal numbers from each digit
        selected_indices = []
        np.random.seed(42)  # For reproducible results
        for digit in range(10):
            selected = np.random.choice(digit_indices[digit], keep_per_digit, replace=False)
            selected_indices.extend(selected)
        
        # Sort indices to maintain some order
        selected_indices.sort()
        
        # Filter images and labels
        images = images[selected_indices]
        labels = labels[selected_indices]
        
        print(f"Selected {len(images)} images total")
    
    print(f"Processing {len(images)} images...")
    feature_data = []
    
    for i in range(len(images)):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(images)} images...")
        
        features = extract_features(images[i])
        # Add label as first column
        row = [labels[i]] + features
        feature_data.append(row)
    
    # Create DataFrame
    df_columns = ['label'] + columns
    df = pd.DataFrame(feature_data, columns=df_columns)
    
    # Save to CSV with appropriate filename
    if omit_percentage > 0:
        csv_filename = f"mnist_features_{100-omit_percentage}percent.csv"
    else:
        csv_filename = "mnist_features_complete.csv"
    
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ Features extracted and saved to '{csv_filename}'")
    print(f"Dataset shape: {df.shape}")
    print(f"Distribution by digit:")
    print(df['label'].value_counts().sort_index())
    print(f"Columns: {list(df.columns)}")