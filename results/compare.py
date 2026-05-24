import numpy as np
import cv2
import sys
import os

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
    # Adjust this threshold if your valid depth can be very close to 0
    valid_mask = (map1 > 0) & (map2 > 0)

    if not np.any(valid_mask):
        raise ValueError("No valid overlapping pixels found (all pixels are 0 or invalid).")

    # Extract valid pixels
    d1_valid = map1[valid_mask]
    d2_valid = map2[valid_mask]

    # Calculate errors
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(d1_valid - d2_valid))
    
    # MSE: Mean Squared Error
    mse = np.mean((d1_valid - d2_valid) ** 2)
    
    # RMSE: Root Mean Squared Error (often more interpretable)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Replace these with your actual file paths
    path1 = "CPU_depth_map.png"
    path2 = "depth_map_fast_convol.png"

    print(f"--- Depth Map Error Analysis ---")
    print(f"Comparing: {path1} vs {path2}\n")

    try:
        # 1. Load images
        depth1 = load_depth_map(path1)
        depth2 = load_depth_map(path2)

        # 2. Calculate metrics
        mae, mse, rmse = calculate_depth_error(depth1, depth2)

        # 3. Output results
        print("\n--- Results ---")
        print(f"Valid Pixels Analyzed: {np.sum((depth1 > 0) & (depth2 > 0))}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Squared Error (MSE):  {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure both images are the same resolution and contain valid depth data (non-zero).")
