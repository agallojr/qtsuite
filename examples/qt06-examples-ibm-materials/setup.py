"""
help install dependencies
"""

import os
import subprocess
import re

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

# Install torch first to satisfy build dependencies for torch-scatter
print("Installing torch first to satisfy build dependencies...", flush=True)
subprocess.check_call(['uv', 'pip', 'install', 'torch==2.3.0'])

# Parse requirements from file
req_file = os.path.join(REPO_DIR, '.', 'requirements.txt')

# Install all requirements directly with uv
print("Installing all requirements...", flush=True)
subprocess.check_call(['uv', 'pip', 'install', '-r', req_file, "--no-build-isolation"])

# Read and parse requirements.txt into extra_depends
def parse_requirements(reqfile):
    """Parse requirements.txt file and return list of dependencies"""
    requirements = []
    if os.path.exists(reqfile):
        with open(reqfile, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

# Parse requirements from the materials requirements.txt
requirements_from_file = parse_requirements(req_file)

extra_depends = [
    "lwfm @ git+https://github.com/lwfm-proj/lwfm@develop",
    "toml",
    "setuptools>=42",
    "triton==3.4.0",
    "torch==2.8.0",
    "torch-scatter==2.1.2",
    "torchvision"
] + requirements_from_file

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu


extra_depends.sort()
deps_dict = {}
# Combine and deduplicate dependencies, keeping last occurrence of each token
# Parse each dependency and keep last occurrence
for dep in extra_depends:
    # Extract token (package name) from dependency string
    if ' @ ' in dep:  # Handle git URLs
        token = dep.split(' @ ')[0].strip()
    else:
        # Handle standard version specifiers (==, >=, <=, >, <, !=, ~=)
        token = re.split(r'[><=!~]', dep)[0].strip()
    deps_dict[token] = dep

# Sort by token and get final dependency list
final_depends = deps_dict.values()


# Dynamic dependency configuration
setup(
    packages=['qt06_examples_ibm_materials'],
    package_dir={'': '.'},
    install_requires=final_depends
)
