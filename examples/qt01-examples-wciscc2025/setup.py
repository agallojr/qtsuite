"""
help install dependencies
"""

import os
import subprocess
from setuptools import setup

# Clone the WCISCC2025 repository if it doesn't exist
repo_url = "https://github.com/olcf/wciscc2025"
repo_dir = "wciscc2025"
if not os.path.exists(repo_dir):
    print(f"Cloning {repo_url}...")
    subprocess.check_call(["git", "clone", repo_url])
    print(f"Successfully cloned {repo_url} to {repo_dir}")
else:
    print(f"Found existing {repo_dir} directory")

subprocess.check_call(['uv', 'pip', 'install', '-r', 'wciscc2025/qlsa/requirements.txt'])

# Parse requirements from file
req_file = os.path.join('wciscc2025', 'qlsa', 'requirements.txt')
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

dependencies.append(extra_depends)

# Dynamic dependency configuration
setup(
    packages=['qt01_examples_wciscc2025'],
    package_dir={'': '.'},
    install_requires=dependencies
)
