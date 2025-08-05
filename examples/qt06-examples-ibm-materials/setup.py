"""
help install dependencies
"""

import os
import sys
import subprocess
from setuptools import setup

# Clone the materials repository if it doesn't exist
REPO_URL = "https://github.com/IBM/materials"
REPO_DIR = "materials"
if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL}...")
    subprocess.check_call(["git", "clone", REPO_URL])
    print(f"Successfully cloned {REPO_URL} to {REPO_DIR}", flush=True)
else:
    print(f"Found existing {REPO_DIR} directory", flush=True)

# Parse requirements from file
req_file = os.path.join(REPO_DIR, '.', 'requirements.txt')
dependencies = []

if os.path.exists(req_file):
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                dependencies.append(line)
    print(f"Found {len(dependencies)} dependencies in {req_file}", flush=True)
else:
    print(f"Warning: {req_file} not found!", flush=True)

pre_depends = [
    "setuptools",
]

pre_depends.extend(dependencies)
dependencies = pre_depends

post_depends = [
    "lwfm @ git+https://github.com/lwfm-proj/lwfm@develop",
    "toml",
]
dependencies.extend(post_depends)
print(dependencies, flush=True)

# Write dependencies to file
output_file = "all_dependencies.txt"
with open(output_file, 'w') as f:
    for dep in dependencies:
        f.write(f"{dep}\n")
print(f"Written {len(dependencies)} dependencies to {output_file}")

# Dynamic dependency configuration
setup(
    packages=['qt06_examples_ibm_materials'],
    package_dir={'': '.'},
    install_requires=dependencies
)
