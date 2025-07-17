import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize

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
    """Apply proper skeletonization using scikit-image"""
    # Use scikit-image's skeletonization
    skeleton = skeletonize(binary)
    return skeleton

def calculate_writing_direction(binary):
    """Calculate writing direction for global and quadrants"""
    skeleton = basic_thinning(binary).astype(np.uint8) * 255
    grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
    
    def get_dominant_angle(gx, gy):
        """Get dominant angle from gradients"""
        if np.sum(np.abs(gx)) == 0 and np.sum(np.abs(gy)) == 0:
            return 0.0
        angles = np.arctan2(gy, gx)
        angles_deg = np.degrees(angles)
        # Get most common angle
        hist, bins = np.histogram(angles_deg, bins=36, range=(-180, 180))
        dominant_bin = np.argmax(hist)
        return (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
    
    # Global direction
    global_angle = get_dominant_angle(grad_x, grad_y)
    
    # Quadrant directions
    h, w = grad_x.shape
    mid_h, mid_w = h // 2, w // 2
    
    # Top-left
    tl_angle = get_dominant_angle(grad_x[:mid_h, :mid_w], grad_y[:mid_h, :mid_w])
    # Top-right
    tr_angle = get_dominant_angle(grad_x[:mid_h, mid_w:], grad_y[:mid_h, mid_w:])
    # Bottom-left
    bl_angle = get_dominant_angle(grad_x[mid_h:, :mid_w], grad_y[mid_h:, :mid_w])
    # Bottom-right
    br_angle = get_dominant_angle(grad_x[mid_h:, mid_w:], grad_y[mid_h:, mid_w:])
    
    return global_angle, tl_angle, tr_angle, bl_angle, br_angle, skeleton

def visualize_writing_direction(image, label, index):
    """Visualize writing direction analysis"""
    binary = image < 50  # threshold for dark pixels
    global_angle, tl_angle, tr_angle, bl_angle, br_angle, skeleton = calculate_writing_direction(binary)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'Original Image (Label: {label})')
    axes[0, 0].axis('off')
    
    # Binary image
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Image')
    axes[0, 1].axis('off')
    
    # Skeleton
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('Skeleton')
    axes[0, 2].axis('off')
    
    # Quadrants with angles
    h, w = image.shape
    mid_h, mid_w = h // 2, w // 2
    
    # Create quadrant visualization
    quad_img = image.copy()
    
    # Draw quadrant lines
    axes[1, 0].imshow(quad_img, cmap='gray')
    axes[1, 0].axhline(y=mid_h, color='red', linewidth=2, alpha=0.7)
    axes[1, 0].axvline(x=mid_w, color='red', linewidth=2, alpha=0.7)
    
    # Add angle text for each quadrant
    axes[1, 0].text(mid_w//2, mid_h//2, f'TL: {tl_angle:.1f}°', 
                    ha='center', va='center', color='yellow', fontsize=10, fontweight='bold')
    axes[1, 0].text(mid_w + mid_w//2, mid_h//2, f'TR: {tr_angle:.1f}°', 
                    ha='center', va='center', color='yellow', fontsize=10, fontweight='bold')
    axes[1, 0].text(mid_w//2, mid_h + mid_h//2, f'BL: {bl_angle:.1f}°', 
                    ha='center', va='center', color='yellow', fontsize=10, fontweight='bold')
    axes[1, 0].text(mid_w + mid_w//2, mid_h + mid_h//2, f'BR: {br_angle:.1f}°', 
                    ha='center', va='center', color='yellow', fontsize=10, fontweight='bold')
    
    axes[1, 0].set_title(f'Quadrant Angles (Global: {global_angle:.1f}°)')
    axes[1, 0].axis('off')
    
    # Direction vectors visualization
    axes[1, 1].imshow(skeleton, cmap='gray')
    
    # Draw direction arrows for each quadrant
    arrow_length = 8
    arrow_props = dict(arrowstyle='->', lw=2, color='red')
    
    # Top-left arrow
    x1, y1 = mid_w//2, mid_h//2
    dx1 = arrow_length * np.cos(np.radians(tl_angle))
    dy1 = -arrow_length * np.sin(np.radians(tl_angle))  # negative for image coordinates
    axes[1, 1].annotate('', xy=(x1+dx1, y1+dy1), xytext=(x1, y1), arrowprops=arrow_props)
    
    # Top-right arrow
    x2, y2 = mid_w + mid_w//2, mid_h//2
    dx2 = arrow_length * np.cos(np.radians(tr_angle))
    dy2 = -arrow_length * np.sin(np.radians(tr_angle))
    axes[1, 1].annotate('', xy=(x2+dx2, y2+dy2), xytext=(x2, y2), arrowprops=arrow_props)
    
    # Bottom-left arrow
    x3, y3 = mid_w//2, mid_h + mid_h//2
    dx3 = arrow_length * np.cos(np.radians(bl_angle))
    dy3 = -arrow_length * np.sin(np.radians(bl_angle))
    axes[1, 1].annotate('', xy=(x3+dx3, y3+dy3), xytext=(x3, y3), arrowprops=arrow_props)
    
    # Bottom-right arrow
    x4, y4 = mid_w + mid_w//2, mid_h + mid_h//2
    dx4 = arrow_length * np.cos(np.radians(br_angle))
    dy4 = -arrow_length * np.sin(np.radians(br_angle))
    axes[1, 1].annotate('', xy=(x4+dx4, y4+dy4), xytext=(x4, y4), arrowprops=arrow_props)
    
    # Draw quadrant lines
    axes[1, 1].axhline(y=mid_h, color='blue', linewidth=1, alpha=0.5)
    axes[1, 1].axvline(x=mid_w, color='blue', linewidth=1, alpha=0.5)
    
    axes[1, 1].set_title('Direction Vectors')
    axes[1, 1].axis('off')
    
    # Summary text
    summary_text = f"""
    Global Angle: {global_angle:.1f}°
    
    Quadrant Angles:
    Top-Left: {tl_angle:.1f}°
    Top-Right: {tr_angle:.1f}°
    Bottom-Left: {bl_angle:.1f}°
    Bottom-Right: {br_angle:.1f}°
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Writing Direction Summary')
    
    plt.tight_layout()
    plt.show()

# Load the data
images = load_images(IMAGE_FILE)
labels = load_labels(LABEL_FILE)

# Visualize writing direction for a specific image
image_index = 2  # Change this number to see different images (0-59999)
visualize_writing_direction(images[image_index], labels[image_index], image_index)