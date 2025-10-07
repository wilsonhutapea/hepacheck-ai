import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt # Added for debugging visualization
import eye_detection_module_v3 as edm

def extract_sclera_rgb(image_path, jaundice_category, debug=False):
    """
    Extract median RGB values from sclera region of eye image and optionally visualize the process.
    
    Args:
        image_path (str): path to the image file
        jaundice_category (int): 0, 1, or 2 (from doctor's rating)
        debug (bool): If True, displays images and extracted colors using matplotlib.
    
    Returns:
        tuple: (red, green, blue, category) or None if processing fails
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ⚠️ Cannot read image: {image_path}")
            return None
        
        # Detect eyes using multi-stage detection
        detection_result = edm.multi_stage_detection(image)
        
        if detection_result is None or not detection_result.get('success'):
            print(f"  ⚠️ Eye detection failed: {image_path}")
            return None
        
        # Process both eyes if available
        rgb_values = []
        
        # A dictionary to hold data for left and right eyes for easy iteration
        eye_data_to_process = {
            'Left': (detection_result.get('left_eye'), detection_result.get('left_mask')),
            'Right': (detection_result.get('right_eye'), detection_result.get('right_mask'))
        }

        for eye_side, (eye_image, eye_mask) in eye_data_to_process.items():
            if eye_image is not None and eye_mask is not None:
                # Convert the detected eye (which is in BGR) to RGB for processing and display
                eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
                sclera_pixels = eye_rgb[eye_mask > 0]
                
                if len(sclera_pixels) > 0:
                    median_rgb = np.median(sclera_pixels, axis=0)
                    rgb_values.append(median_rgb)

                    # --- DEBUG VISUALIZATION BLOCK ---
                    if debug:
                        # Create a color swatch image from the median RGB value
                        color_swatch = np.full((100, 100, 3), median_rgb, dtype=np.uint8)
                        
                        # Set up the plot
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        fig.suptitle(f"Debug: {Path(image_path).name} - {eye_side} Eye", fontsize=14)

                        # Display detected eye
                        axes[0].imshow(eye_rgb)
                        axes[0].set_title("1. Detected Eye")
                        axes[0].axis('off')

                        # Display sclera mask
                        axes[1].imshow(eye_mask, cmap='gray')
                        axes[1].set_title("2. Sclera Mask")
                        axes[1].axis('off')

                        # Display extracted color
                        axes[2].imshow(color_swatch)
                        axes[2].set_title(f"3. Extracted Color\nRGB: {tuple(int(c) for c in median_rgb)}")
                        axes[2].axis('off')
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
                        plt.show()
                    # --- END DEBUG BLOCK ---
        
        if not rgb_values:
            print(f"  ⚠️ No sclera pixels found: {image_path}")
            return None
        
        # Average RGB from both eyes (if both detected)
        avg_rgb = np.mean(rgb_values, axis=0)
        red, green, blue = avg_rgb
        
        return (int(red), int(green), int(blue), jaundice_category)
    
    except Exception as e:
        print(f"  ❌ Error processing {image_path}: {str(e)}")
        return None


def create_jaundice_dataset(base_folder, output_csv='jaundice_dataset.csv', debug=False):
    """
    Process images from folders 0, 1, 2 and create CSV dataset
    
    Args:
        base_folder (str): path to folder containing subfolders 0, 1, 2
        output_csv (str): output CSV filename
        debug (bool): If True, enables visualization for each image.
    
    Returns:
        pandas DataFrame with the dataset
    """
    print("=" * 70)
    print("JAUNDICE DATASET GENERATOR")
    print("=" * 70)
    
    dataset = []
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Process each category folder
    for category in [0, 1, 2]:
        folder_path = Path(base_folder) / str(category)
        
        if not folder_path.exists():
            print(f"\n⚠️  Warning: Folder '{folder_path}' does not exist. Skipping...")
            continue
        
        print(f"\n📁 Processing Category {category} folder: {folder_path}")
        print("-" * 70)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(folder_path.glob(f'*{ext}')))
            image_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images in category {category}")
        
        for img_path in image_files:
            total_images += 1
            print(f"\n  Processing: {img_path.name}")
            
            # Extract RGB values, passing the debug flag
            result = extract_sclera_rgb(str(img_path), category, debug=debug)
            
            if result is not None:
                dataset.append(result)
                processed_images += 1
                red, green, blue, cat = result
                print(f"  ✓ Success: R={red}, G={green}, B={blue}, Category={cat}")
            else:
                failed_images += 1
    
    # Create DataFrame and print summary (code is unchanged from here)
    print("\n" + "=" * 70)
    print("DATASET CREATION SUMMARY")
    print("=" * 70)
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {failed_images}")
    
    if len(dataset) == 0:
        print("\n❌ No data extracted. CSV file not created.")
        return None
    
    df = pd.DataFrame(dataset, columns=['RedValue', 'GreenValue', 'BlueValue', 'JaundiceCategory'])
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Dataset saved to: {output_csv}")
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION:")
    print("-" * 70)
    category_counts = df['JaundiceCategory'].value_counts().sort_index()
    for cat, count in category_counts.items():
        category_name = ['No/Low Risk', 'Mild Jaundice', 'Severe Jaundice'][cat]
        print(f"  Category {cat} ({category_name}): {count} images")
    
    print("\n" + "-" * 70)
    print("RGB VALUE STATISTICS BY CATEGORY:")
    print("-" * 70)
    for cat in sorted(df['JaundiceCategory'].unique()):
        category_name = ['No/Low Risk', 'Mild Jaundice', 'Severe Jaundice'][cat]
        cat_data = df[df['JaundiceCategory'] == cat]
        print(f"\nCategory {cat} ({category_name}):")
        print(f"  Red   - Mean: {cat_data['RedValue'].mean():.1f}, Std: {cat_data['RedValue'].std():.1f}")
        print(f"  Green - Mean: {cat_data['GreenValue'].mean():.1f}, Std: {cat_data['GreenValue'].std():.1f}")
        print(f"  Blue  - Mean: {cat_data['BlueValue'].mean():.1f}, Std: {cat_data['BlueValue'].std():.1f}")
    
    print("\n" + "-" * 70)
    print("SAMPLE DATA (first 10 rows):")
    print("-" * 70)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    
    return df


def create_jaundice_dataset_from_filename(base_folder, output_csv='jaundice_dataset.csv', debug=False):
    """
    Process images and extract jaundice category from filename prefix
    
    Args:
        base_folder (str): path to folder containing all images
        output_csv (str): output CSV filename
        debug (bool): If True, enables visualization for each image.
    
    Returns:
        pandas DataFrame with the dataset
    """
    print("=" * 70)
    print("JAUNDICE DATASET GENERATOR (Filename-based Category)")
    print("=" * 70)
    
    dataset = []
    total_images = 0
    processed_images = 0
    failed_images = 0
    skipped_images = 0
    
    folder_path = Path(base_folder)
    
    if not folder_path.exists():
        print(f"\n❌ Error: Folder '{folder_path}' does not exist.")
        return None
    
    print(f"\n📁 Processing folder: {folder_path}")
    print("-" * 70)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(folder_path.glob(f'*{ext}')))
        image_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    for img_path in image_files:
        total_images += 1
        filename = img_path.name
        
        first_char = filename[0]
        if first_char not in ['0', '1', '2']:
            print(f"\n  ⚠️  Skipping {filename}: First character '{first_char}' is not 0, 1, or 2")
            skipped_images += 1
            continue
        
        category = int(first_char)
        print(f"\n  Processing: {filename} (Category: {category})")
        
        # Extract RGB values, passing the debug flag
        result = extract_sclera_rgb(str(img_path), category, debug=debug)
        
        if result is not None:
            dataset.append(result)
            processed_images += 1
            red, green, blue, cat = result
            print(f"  ✓ Success: R={red}, G={green}, B={blue}, Category={cat}")
        else:
            failed_images += 1
    
    # Create DataFrame and print summary (code is unchanged from here)
    print("\n" + "=" * 70)
    print("DATASET CREATION SUMMARY")
    print("=" * 70)
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Skipped (invalid filename): {skipped_images}")
    print(f"Failed (processing error): {failed_images}")
    
    if len(dataset) == 0:
        print("\n❌ No data extracted. CSV file not created.")
        return None
    
    df = pd.DataFrame(dataset, columns=['RedValue', 'GreenValue', 'BlueValue', 'JaundiceCategory'])
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Dataset saved to: {output_csv}")
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # ... (Statistics and sample data display remains the same as the function above)
    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION:")
    print("-" * 70)
    category_counts = df['JaundiceCategory'].value_counts().sort_index()
    for cat, count in category_counts.items():
        category_name = ['No/Low Risk', 'Mild Jaundice', 'Severe Jaundice'][cat]
        print(f"  Category {cat} ({category_name}): {count} images")
    
    print("\n" + "-" * 70)
    print("RGB VALUE STATISTICS BY CATEGORY:")
    print("-" * 70)
    for cat in sorted(df['JaundiceCategory'].unique()):
        category_name = ['No/Low Risk', 'Mild Jaundice', 'Severe Jaundice'][cat]
        cat_data = df[df['JaundiceCategory'] == cat]
        print(f"\nCategory {cat} ({category_name}):")
        print(f"  Red   - Mean: {cat_data['RedValue'].mean():.1f}, Std: {cat_data['RedValue'].std():.1f}")
        print(f"  Green - Mean: {cat_data['GreenValue'].mean():.1f}, Std: {cat_data['GreenValue'].std():.1f}")
        print(f"  Blue  - Mean: {cat_data['BlueValue'].mean():.1f}, Std: {cat_data['BlueValue'].std():.1f}")
    
    print("\n" + "-" * 70)
    print("SAMPLE DATA (first 10 rows):")
    print("-" * 70)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    
    return df


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    
    print("Choose processing method:")
    print("1. Images organized in folders: 0/, 1/, 2/")
    print("2. Images in single folder with category prefix in filename (e.g., '2_firstphoto.jpg')")
    
    method = input("\nEnter method (1 or 2): ").strip()
    
    # NEW: Ask user if they want to enable debug mode
    debug_input = input("Enable debug mode to see images for each step? (y/n): ").strip().lower()
    debug_mode = (debug_input == 'y')
    
    if debug_mode:
        print("\n🔎 Debug mode ENABLED. Plots will be shown for each processed eye.")
    
    df = None
    if method == "1":
        base_folder = input("Enter path to base folder containing 0, 1, 2 subfolders: ").strip()
        output_csv = input("Enter output CSV filename (default: jaundice_dataset.csv): ").strip() or 'jaundice_dataset.csv'
        
        df = create_jaundice_dataset(base_folder, output_csv, debug=debug_mode)
    
    elif method == "2":
        base_folder = input("Enter path to folder containing all images: ").strip()
        output_csv = input("Enter output CSV filename (default: jaundice_dataset.csv): ").strip() or 'jaundice_dataset.csv'
        
        df = create_jaundice_dataset_from_filename(base_folder, output_csv, debug=debug_mode)
    
    else:
        print("Invalid method selected.")
    
    if df is not None:
        print("\n✅ Dataset creation complete!")
        print(f"You can now use '{output_csv}' for training your jaundice detection model.")