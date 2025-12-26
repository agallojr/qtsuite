"""
Utility functions for qp4p helpers.
"""
import argparse
import numpy as np
from PIL import Image


def load_or_generate_image(image_path: str, size: int) -> np.ndarray:
    """Load image from file or generate sample data if file doesn't exist."""
    if image_path:
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((size, size))
            return np.array(img).flatten().astype(float)
        except FileNotFoundError:
            print(f"File '{image_path}' not found, generating sample data.")
    
    # Generate sample gradient pattern (brighter on right side)
    pattern = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            pattern[row, col] = 50 + (col / (size - 1)) * 150  # Gradient left to right
    return pattern.flatten().astype(float)


def validate_power_of_2(value: str) -> int:
    """Validate that size is a power of 2."""
    ivalue = int(value)
    if ivalue <= 0 or (ivalue & (ivalue - 1)) != 0:
        raise argparse.ArgumentTypeError(f"{value} is not a power of 2")
    return ivalue
