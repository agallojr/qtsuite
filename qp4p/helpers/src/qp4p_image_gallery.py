#!/usr/bin/env python3
"""
Gallery visualization for image sweep results.

Collects all PNG visualizations from a sweep and displays them in a navigable window.
Uses matplotlib for portability (works across macOS updates that break Tkinter/PIL).

Usage:
    python helpers/src/qp4p_image_gallery.py <final_postproc_json>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button


def create_gallery(data, output_file=None, display=True):
    """
    Create an image gallery with navigation buttons.
    
    Args:
        data: Dictionary containing final postproc context
        output_file: Optional path to save the gallery (not used)
        display: Whether to display the gallery
    """
    case_dirs = data.get("case_dirs", [])
    
    # Find all PNG files in case directories
    png_files = []
    for case_dir_str in case_dirs:
        case_dir = Path(case_dir_str)
        for png_file in case_dir.glob("*.png"):
            png_files.append(png_file)
    
    if not png_files:
        print("No PNG files found in case directories")
        return
    
    # Sort by parent directory name then filename for consistent ordering
    png_files.sort(key=lambda p: (p.parent.name, p.name))
    
    if not display:
        print(f"Found {len(png_files)} images (display disabled)")
        return
    
    print(f"Opening gallery with {len(png_files)} images...")
    
    # State
    current_index = [0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.15)
    
    def show_image(index):
        """Display image at given index."""
        ax.clear()
        png_file = png_files[index]
        img = mpimg.imread(str(png_file))
        ax.imshow(img)
        ax.set_title(f"{png_file.parent.name}/{png_file.name}\n({index + 1} of {len(png_files)})", 
                    fontsize=11)
        ax.axis('off')
        fig.canvas.draw_idle()
    
    def next_image(event):
        current_index[0] = (current_index[0] + 1) % len(png_files)
        show_image(current_index[0])
    
    def prev_image(event):
        current_index[0] = (current_index[0] - 1) % len(png_files)
        show_image(current_index[0])
    
    def on_key(event):
        if event.key in ('right', 'n', 'down'):
            next_image(event)
        elif event.key in ('left', 'p', 'up'):
            prev_image(event)
        elif event.key in ('q', 'escape'):
            plt.close(fig)
    
    # Add navigation buttons
    ax_prev = plt.axes([0.3, 0.02, 0.15, 0.05])
    ax_next = plt.axes([0.55, 0.02, 0.15, 0.05])
    btn_prev = Button(ax_prev, '← Previous (Left/P)')
    btn_next = Button(ax_next, 'Next (Right/N) →')
    btn_prev.on_clicked(prev_image)
    btn_next.on_clicked(next_image)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show first image
    show_image(0)
    
    # Set window title
    fig.canvas.manager.set_window_title(f"Image Gallery ({len(png_files)} images)")
    
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python helpers/src/qp4p_image_gallery.py <final_postproc_json>")
        sys.exit(1)
    
    # Print invocation command for easy re-opening
    postproc_json = Path(sys.argv[1])
    print(f"Image gallery command: python helpers/src/qp4p_image_gallery.py {postproc_json}")
    
    # Load final postproc context from JSON file
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    create_gallery(context, display=True)


if __name__ == "__main__":
    main()
