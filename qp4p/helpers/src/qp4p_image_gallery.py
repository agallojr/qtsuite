#!/usr/bin/env python3
"""
Gallery visualization for image sweep results.

Collects all PNG visualizations from a sweep and displays them in a scrollable window.

Usage:
    python -m qp4p_image_gallery <final_postproc_json>
"""

import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def create_gallery(data, output_file=None, display=True):
    """
    Create an image gallery with file list and viewer.
    
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
    
    # Sort by filename for consistent ordering
    png_files.sort(key=lambda p: p.name)
    
    if not display:
        print(f"Found {len(png_files)} images (display disabled)")
        return
    
    # Create main window
    root = tk.Tk()
    root.title(f"Image Gallery ({len(png_files)} images)")
    root.geometry("1400x900")
    
    # Create main container with two panes
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)
    
    # Left pane: File list
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, weight=0)
    
    # Set minimum width for left pane
    left_frame.pack_propagate(False)
    left_frame.config(width=100)
    
    # File list with scrollbar
    list_label = ttk.Label(left_frame, text="Images:", font=('Arial', 10, 'bold'))
    list_label.pack(pady=5)
    
    list_frame = ttk.Frame(left_frame)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    list_scrollbar = ttk.Scrollbar(list_frame)
    list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    file_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, font=('Arial', 9))
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    list_scrollbar.config(command=file_listbox.yview)
    
    # Populate file list
    for png_file in png_files:
        display_name = f"{png_file.parent.name}"
        file_listbox.insert(tk.END, display_name)
    
    # Right pane: Image viewer
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, weight=1)
    
    # Info label
    info_label = ttk.Label(right_frame, text="Select an image from the list", font=('Arial', 10))
    info_label.pack(pady=5)
    
    # Image canvas with scrollbars
    canvas_frame = ttk.Frame(right_frame)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
    
    image_canvas = tk.Canvas(canvas_frame, bg='white',
                            xscrollcommand=h_scrollbar.set,
                            yscrollcommand=v_scrollbar.set)
    
    h_scrollbar.config(command=image_canvas.xview)
    v_scrollbar.config(command=image_canvas.yview)
    
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Navigation buttons
    nav_frame = ttk.Frame(right_frame)
    nav_frame.pack(pady=5)
    
    prev_btn = ttk.Button(nav_frame, text="← Previous")
    prev_btn.pack(side=tk.LEFT, padx=5)
    
    next_btn = ttk.Button(nav_frame, text="Next →")
    next_btn.pack(side=tk.LEFT, padx=5)
    
    # State variables
    current_photo = [None]  # List to allow modification in nested function
    
    def load_image(index):
        if 0 <= index < len(png_files):
            png_file = png_files[index]
            
            # Update info label
            info_text = f"{png_file.parent.name} - {png_file.name}"
            info_label.config(text=info_text)
            
            # Load and display image
            img = Image.open(png_file)
            photo = ImageTk.PhotoImage(img)
            current_photo[0] = photo  # Keep reference
            
            # Clear canvas and display image
            image_canvas.delete("all")
            image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            image_canvas.config(scrollregion=(0, 0, img.width, img.height))
            
            # Update listbox selection
            file_listbox.selection_clear(0, tk.END)
            file_listbox.selection_set(index)
            file_listbox.see(index)
    
    def on_select(event):
        selection = file_listbox.curselection()
        if selection:
            load_image(selection[0])
    
    def prev_image():
        selection = file_listbox.curselection()
        if selection and selection[0] > 0:
            load_image(selection[0] - 1)
    
    def next_image():
        selection = file_listbox.curselection()
        if selection and selection[0] < len(png_files) - 1:
            load_image(selection[0] + 1)
    
    # Bind events
    file_listbox.bind('<<ListboxSelect>>', on_select)
    prev_btn.config(command=prev_image)
    next_btn.config(command=next_image)
    
    # Keyboard shortcuts
    root.bind('<Left>', lambda e: prev_image())
    root.bind('<Right>', lambda e: next_image())
    root.bind('<Up>', lambda e: prev_image())
    root.bind('<Down>', lambda e: next_image())
    
    # Load first image
    if png_files:
        load_image(0)
    
    root.mainloop()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/image_gallery.py <final_postproc_json>")
        sys.exit(1)
    
    # Load final postproc context from JSON file
    postproc_json = Path(sys.argv[1])
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    create_gallery(context, display=True)


if __name__ == "__main__":
    main()
