import numpy as np
import cv2
import sys
import os
import csv

def load_depth_map(filepath):
    """
    Loads a depth map image and converts it to a float32 numpy array.
    Handles 8-bit, 16-bit, and floating point images.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # cv2.IMREAD_UNCHANGED preserves the original bit depth
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image: {filepath}")

    print(f"Loaded {filepath}: Shape={img.shape}, Dtype={img.dtype}, Range=[{img.min()}, {img.max()}]")

    # Convert to float32 for calculation
    depth = img.astype(np.float32)

    # Normalize if the image is 8-bit (0-255) to typical depth units if needed.
    if img.dtype == np.uint8:
        print("  -> Detected 8-bit image. Normalizing to 0.0-1.0 range.")
        depth /= 255.0
    return depth

def calculate_depth_error(map1, map2):
    """
    Calculates MAE, MSE, and RMSE between two depth maps.
    Ignores pixels where either map is 0 (assuming 0 is 'invalid' depth).
    """
    if map1.shape != map2.shape:
        raise ValueError(f"Image shapes do not match: {map1.shape} vs {map2.shape}")

    # Create a mask for valid pixels (where depth > 0)
    valid_mask = (map1 > 0) & (map2 > 0)

    if not np.any(valid_mask):
        raise ValueError("No valid overlapping pixels found (all pixels are 0 or invalid).")

    # Extract valid pixels
    d1_valid = map1[valid_mask]
    d2_valid = map2[valid_mask]

    # Calculate errors
    mae = np.mean(np.abs(d1_valid - d2_valid))
    mse = np.mean((d1_valid - d2_valid) ** 2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

def write_metrics_to_csv(filename, path2_name, mae, mse, rmse, valid_pixels):
    """
    Appends the results to a CSV file. Creates a header if the file doesn't exist.
    """
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["Comparison", "Valid Pixels", "MAE", "MSE", "RMSE"])
        
        writer.writerow([path2_name, valid_pixels, f"{mae:.6f}", f"{mse:.6f}", f"{rmse:.6f}"])

if __name__ == "__main__":
    # --- CONFIGURATION ---
    path1 = "CPU_depth_map.png"
    csv_filename = "depth_error_metrics.csv"
    
    # Get the current directory path
    current_dir = os.getcwd()
    
    # List all entries in the directory
    for path2 in os.listdir(current_dir):
        # Skip the reference file itself and non-png files
        if path2 == path1 or not path2.endswith(".png"):
            continue

        print(f"--- Depth Map Error Analysis ---")
        print(f"Comparing: {path1} vs {path2}\n")

        try:
            # 1. Load images
            depth1 = load_depth_map(path1)
            depth2 = load_depth_map(path2)

            # 2. Calculate metrics
            mae, mse, rmse = calculate_depth_error(depth1, depth2)
            
            # Count valid pixels for the report
            valid_pixels = int(np.sum((depth1 > 0) & (depth2 > 0)))

            # 3. Output results to console
            print("\n--- Results ---")
            print(f"Valid Pixels Analyzed: {valid_pixels}")
            print(f"Mean Absolute Error (MAE): {mae:.6f}")
            print(f"Mean Squared Error (MSE):  {mse:.6f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

            # 4. Write to CSV
            write_metrics_to_csv(csv_filename, path2, mae, mse, rmse, valid_pixels)
            print(f"Results appended to {csv_filename}")

        except Exception as e:
            print(f"Error: {e}")
            print("Tip: Ensure both images are the same resolution and contain valid depth data (non-zero).")