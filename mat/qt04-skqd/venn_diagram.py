"""
Generate a 3-way Venn diagram for AI DevOps, Quantum, and Workflow
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Define circle parameters (x, y, radius)
circles = [
    {'center': (0, 0.5), 'radius': 1.5, 'color': '#FF5733', 'alpha': 0.5, 'label': 'AI DevOps'},
    {'center': (1.5, 0.5), 'radius': 1.5, 'color': '#3498DB', 'alpha': 0.5, 'label': 'Quantum\nComputing'},
    {'center': (0.75, -0.8), 'radius': 1.5, 'color': '#9B59B6', 'alpha': 0.5, 'label': '"The Workflow is\nthe App"'},
]

# Draw circles
for circle in circles:
    c = Circle(circle['center'], circle['radius'], 
               color=circle['color'], alpha=circle['alpha'], 
               edgecolor='black', linewidth=2)
    ax.add_patch(c)

# Add labels inside circles with center alignment
ax.text(circles[0]['center'][0] - 0.6, circles[0]['center'][1] + 0.7, 
        circles[0]['label'], fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(circles[1]['center'][0] + 0.6, circles[1]['center'][1] + 0.7, 
        circles[1]['label'], fontsize=13, fontweight='bold', ha='center', va='center')
ax.text(circles[2]['center'][0], circles[2]['center'][1] - 0.7, 
        circles[2]['label'], fontsize=12, fontweight='bold', ha='center', va='center')

# Set axis properties
ax.set_xlim(-2.5, 4)
ax.set_ylim(-3, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# Add title
plt.title('3-Way Venn Diagram', fontsize=20, fontweight='bold', pad=20)

# Save figure
plt.tight_layout()
plt.savefig('venn_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Venn diagram saved as 'venn_diagram.png'")

# Display
plt.show()
