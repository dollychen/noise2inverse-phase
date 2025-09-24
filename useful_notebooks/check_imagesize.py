import os
import cv2

def flag_inconsistent_images(input_dir):
    # Define valid image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # Get list of image files sorted alphanumerically
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    # Read the first image and use its shape as the reference
    first_image_path = os.path.join(input_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    
    if first_image is None:
        print(f"Error: Unable to read the first image: {first_image_path}")
        return
    
    reference_shape = first_image.shape
    print("Reference shape (height, width, channels):", reference_shape)
    
    # Check each image against the reference shape
    inconsistent_images = []
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not read image '{filename}'.")
            inconsistent_images.append(filename)
            continue
        
        if img.shape != reference_shape:
            print(f"Flag: Image '{filename}' has shape {img.shape} but expected {reference_shape}")
            inconsistent_images.append(filename)
    
    if inconsistent_images:
        print("\nInconsistent images found:")
        for fname in inconsistent_images:
            print(" -", fname)
    else:
        print("\nAll images match the reference shape.")

# Example usage:
if __name__ == "__main__":
    input_directory = "/cluster/project7/HiP_CT_Denoise/HiPCT/orig_pag/split_2/1"  # Replace with your directory path
    flag_inconsistent_images(input_directory)