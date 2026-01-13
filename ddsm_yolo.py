import os
import shutil
import cv2
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = 'data'
CSV_DIR = os.path.join(DATA_ROOT, 'csv')
JPEG_DIR = os.path.join(DATA_ROOT, 'jpeg')
OUTPUT_DIR = 'yolo_dataset'

# Class Mapping
CLASS_MAP = {
    'BENIGN': 0,
    'BENIGN_WITHOUT_CALLBACK': 0,
    'MALIGNANT': 1
}

def setup_directories():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def find_image_and_mask_smart(image_dir, csv_path_entry):
    """
    Tries to find the folder by checking ALL UIDs in the CSV path.
    Once the folder is found, it grabs the first .jpg file inside.
    """
    path_parts = csv_path_entry.strip().split('/')
    
    # 1. Identify all parts that look like UIDs
    potential_uids = [p for p in path_parts if p.startswith('1.3.6.1.4.1')]
    
    found_folder = None
    
    # 2. Check which UID exists as a folder
    for uid in potential_uids:
        candidate_path = os.path.join(image_dir, uid)
        if os.path.exists(candidate_path):
            found_folder = candidate_path
            break
            
    if not found_folder:
        return None

    # 3. Find the .jpg inside that folder (recursively)
    for root, dirs, files in os.walk(found_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                return os.path.join(root, file)
                
    return None

def get_yolo_coordinates(mask_path):
    # Read mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    white_pixels = np.sum(mask ==255)
    total_pixels = mask.size

    if white_pixels > (total_pixels * 0.5):
        mask = cv2.bitwise_not(mask)

    # Threshold to ensure binary (0 or 255)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # --- FIX 2: Filter Bad Contours ---
    img_h, img_w = mask.shape
    valid_contour = None
    
    # Sort contours by area (largest first) so we check the main object first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Calculate how much of the image this box covers
        box_area = w * h
        img_area = img_w * img_h
        ratio = box_area / img_area
        
        # If the box covers > 95% of the image, it's the image border/background. Skip it.
        if ratio > 0.95:
            continue 
            
        # If the box is extremely tiny (e.g. < 0.1%), it's just noise/dust. Skip it.
        if ratio < 0.001: 
            continue 
            
        # If we pass those checks, this is likely our tumor
        valid_contour = c
        break 
    
    # If we filtered everything out (e.g. only had a full-image box), return None
    if valid_contour is None:
        return None

    # Calculate final YOLO coords from the valid contour
    x, y, w, h = cv2.boundingRect(valid_contour)

    # Normalize for YOLO (0-1 range)
    img_h, img_w = mask.shape
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    return (x_center, y_center, width, height)

def process_set(csv_filename, split_name):
    print(f"\nProcessing {split_name} set from {csv_filename}...")
    
    csv_path = os.path.join(CSV_DIR, csv_filename)
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_filename} (not found)")
        return

    df = pd.read_csv(csv_path)
    count = 0
    missing = 0

    for idx, row in df.iterrows():
        # 1. Find the actual Image and Mask files
        real_img_path = find_image_and_mask_smart(JPEG_DIR, row['image file path'])
        real_mask_path = find_image_and_mask_smart(JPEG_DIR, row['ROI mask file path'])

        if not real_img_path or not real_mask_path:
            # Only print first few missing errors to avoid spamming console
            if missing < 5: 
                print(f"Missing file for Row {idx}")
                print(f"  CSV Image Path: {row['image file path']}")
                if real_img_path: print("  -> Image FOUND")
                else: print("  -> Image NOT FOUND")
                if real_mask_path: print("  -> Mask FOUND") 
                else: print("  -> Mask NOT FOUND")
            missing += 1
            continue

        # 2. Calculate YOLO BBox from Mask
        yolo_coords = get_yolo_coordinates(real_mask_path)
        if not yolo_coords:
            continue

        # 3. Define Output Filenames
        # Using row index ensures uniqueness even if filenames are duplicate
        file_basename = f"{split_name}_{idx}_{os.path.basename(real_img_path)}"
        dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, file_basename)
        dst_lbl = os.path.join(OUTPUT_DIR, 'labels', split_name, file_basename.replace('.jpg', '.txt'))

        # 4. Save Label
        class_id = CLASS_MAP.get(row['pathology'], 0)
        xc, yc, w, h = yolo_coords
        with open(dst_lbl, 'w') as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # 5. Copy Image
        shutil.copy(real_img_path, dst_img)
        count += 1
        
        if count % 50 == 0:
            print(f"  Processed {count} images...")

    print(f"Completed {split_name}. Created: {count}, Missing/Skipped: {missing}")

# --- RUN ---
setup_directories()
process_set('mass_case_description_train_set.csv', 'train')
process_set('mass_case_description_test_set.csv', 'val')