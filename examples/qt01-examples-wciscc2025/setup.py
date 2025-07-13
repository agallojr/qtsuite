"""
help install dependencies
"""

import os
import sys
import subprocess
from setuptools import setup

# Clone the WCISCC2025 repository if it doesn't exist
REPO_URL = "https://github.com/olcf/wciscc2025"
REPO_DIR = "wciscc2025"
if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL}...")
    subprocess.check_call(["git", "clone", REPO_URL])
    print(f"Successfully cloned {REPO_URL} to {REPO_DIR}")
else:
    print(f"Found existing {REPO_DIR} directory")

# Parse requirements from file
req_file = os.path.join(REPO_DIR, 'qlsa', 'requirements.txt')
dependencies = []

if os.path.exists(req_file):
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                dependencies.append(line)
    print(f"Found {len(dependencies)} dependencies in {req_file}")
else:
    print(f"Warning: {req_file} not found!")

extra_depends = [
    "lwfm @ git+https://github.com/lwfm-proj/lwfm",
    "setuptools",
    "numpy",
    "scipy",
    "toml"
]

dependencies.extend(extra_depends)

# Dynamic dependency configuration
setup(
    packages=['qt01_examples_wciscc2025'],
    package_dir={'': '.'},
    install_requires=dependencies
)
