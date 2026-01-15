#!/usr/bin/env python3
"""
Visualization script for quantum image flip results.

Reads JSON output from image_flip.py and generates visualization plots.

Usage:
    python src/postproc/image_viz.py <stdout_json_file>
    or
    python -m postproc.image_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_image_visualization(data, output_file=None, display=True):
    """
    Create visualization comparing original and quantum-mirrored images.
    
    Args:
        data: Dictionary containing image flip results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract from new standardized JSON structure
    results = data.get("results", {})
    problem = data.get("problem", {})
    config = data.get("config", {})
    
    original_image = np.array(results.get("original_image", []))
    mirrored_image = np.array(results.get("mirrored_image", []))
    fidelity = results.get("fidelity", 0.0)
    
    image_info = problem.get("image", {})
    size = image_info.get("size", 0)
    shots = config.get("shots", 0)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mirrored_image, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"Quantum Mirror (fidelity={fidelity:.3f})", 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add info text
    info_text = f"Size: {size}×{size} | Shots: {shots}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/image_viz.py <postproc_json>")
        sys.exit(1)
    
    # Load postproc context from JSON file
    postproc_json = Path(sys.argv[1])
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    # Process each case directory
    for case_dir_str in context.get("case_dirs", []):
        case_dir = Path(case_dir_str)
        stdout_file = case_dir / "stdout.json"
        
        if not stdout_file.exists():
            print(f"Warning: stdout.json not found in {case_dir}")
            continue
        
        with open(stdout_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Generate output filename based on image config
        problem = data.get("problem", {})
        image_info = problem.get("image", {})
        size = image_info.get("size", 0)
        source = image_info.get("source")
        
        # Clean up source name for filename
        if source == "generated_gradient":
            source_name = "gradient"
        elif source is None or source == "unknown":
            source_name = "unknown"
        else:
            source_name = Path(source).stem
        
        output_file = case_dir / f'image_flip_{size}x{size}_{source_name}.png'
        
        create_image_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
