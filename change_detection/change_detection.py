import cv2
import numpy as np
import os
from pathlib import Path
import argparse


def detect_changes(before_img, after_img, sensitivity=25):
    """
    Detect changes between two aligned images.
    
    Args:
        before_img: Before image (numpy array)
        after_img: After image (numpy array)
        sensitivity: Threshold for change detection (0-255, lower = more sensitive)
    
    Returns:
        change_mask: Binary mask of changes
        highlighted_img: Image with changes highlighted
    """
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    before_blur = cv2.GaussianBlur(before_gray, (5, 5), 0)
    after_blur = cv2.GaussianBlur(after_gray, (5, 5), 0)
    
    # Compute absolute difference
    diff = cv2.absdiff(before_blur, after_blur)
    
    # Apply threshold to get binary mask
    _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to make contours more visible
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    return thresh


def highlight_changes(after_img, change_mask, min_area=100):
    """
    Draw bounding boxes around detected changes.
    
    Args:
        after_img: After image to draw on
        change_mask: Binary mask of changes
        min_area: Minimum contour area to consider
    
    Returns:
        highlighted_img: Image with changes highlighted
    """
    # Create a copy for drawing
    highlighted = after_img.copy()
    
    # Find contours
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around changes
    change_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw rectangle
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), 
                         (0, 0, 255), 2)
            
            # Optionally draw the contour polygon
            cv2.drawContours(highlighted, [contour], -1, (0, 255, 0), 2)
            
            change_count += 1
    
    return highlighted, change_count


def process_image_pair(before_path, after_path, output_dir, sensitivity=25, 
                       min_area=100):
    """
    Process a single pair of before-and-after images.
    
    Args:
        before_path: Path to before image
        after_path: Path to after image
        output_dir: Directory to save output
        sensitivity: Change detection sensitivity
        min_area: Minimum area for change detection
    """
    # Load images
    before_img = cv2.imread(str(before_path))
    after_img = cv2.imread(str(after_path))
    
    if before_img is None or after_img is None:
        print(f"Error loading images: {before_path} or {after_path}")
        return
    
    # Check if images have same dimensions
    if before_img.shape != after_img.shape:
        print(f"Warning: Images have different dimensions. Resizing after image.")
        after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
    
    # Detect changes
    change_mask = detect_changes(before_img, after_img, sensitivity)
    
    # Highlight changes
    highlighted, change_count = highlight_changes(after_img, change_mask, min_area)
    
    # Create output filename
    base_name = Path(before_path).stem
    output_path = Path(output_dir) / f"{base_name}_changes.jpg"
    
    # Save result
    cv2.imwrite(str(output_path), highlighted)
    
    print(f"Processed {base_name}: {change_count} changes detected")
    
    # Optionally save the difference mask
    mask_path = Path(output_dir) / f"{base_name}_mask.jpg"
    cv2.imwrite(str(mask_path), change_mask)


def main():
    parser = argparse.ArgumentParser(
        description='Detect changes between before-and-after images'
    )
    parser.add_argument('input_folder', type=str, 
                       help='Folder containing before-and-after image pairs')
    parser.add_argument('--output', type=str, default='output',
                       help='Output folder for results (default: output)')
    parser.add_argument('--sensitivity', type=int, default=25,
                       help='Change detection sensitivity (0-255, default: 25)')
    parser.add_argument('--min-area', type=int, default=100,
                       help='Minimum area for detected changes (default: 100)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all before images (X.jpg pattern)
    input_path = Path(args.input_folder)
    before_images = sorted([f for f in input_path.glob('*.jpg') 
                           if '~2' not in f.name])
    
    if not before_images:
        print(f"No before images found in {args.input_folder}")
        return
    
    print(f"Found {len(before_images)} image pairs to process")
    
    # Process each pair
    for before_path in before_images:
        # Construct after image path
        base_name = before_path.stem
        after_path = before_path.parent / f"{base_name}~2.jpg"
        
        if not after_path.exists():
            print(f"Warning: After image not found for {before_path.name}")
            continue
        
        process_image_pair(before_path, after_path, output_dir, 
                          args.sensitivity, args.min_area)
    
    print(f"\nProcessing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()