"""
help install dependencies
"""

import os
import subprocess
from setuptools import setup

# Clone the fvm euler 1d repository if it doesn't exist
git_user_key = os.environ.get("GIT_USER_KEY")
if not git_user_key:
    # You must set this environment variable for the installation to work
    raise ValueError("GIT_USER_KEY environment variable is not set - use 'user:key' format")
REPO_URL = os.environ.get("FVM_REPO_URL",
    f"https://{git_user_key}@github.com/mhawwary/fvm_euler_1d_solver")
REPO_DIR = os.environ.get("FVM_REPO_DIR", "fvm_euler_1d_solver")
if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL}...")
    subprocess.check_call(["git", "clone", "-b", "feature/agmods", REPO_URL])
    print(f"Successfully cloned {REPO_URL} to {REPO_DIR}")
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
    print(f"Found existing {REPO_DIR} directory - re-pulling")

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
# notice here no dependency on qiskit - its all in the qlsa requirements.txt, meaning,
# its their dependent version of qiskit etc
req_file = os.path.join(REPO_DIR, 'qlsa', 'requirements.txt')
dependencies = []

if os.path.exists(req_file):
    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                dependencies.append(line)
    print(f"Found {len(dependencies)} dependencies in {req_file}")
else:
    print(f"Warning: {req_file} not found!")

extra_depends = [
    "setuptools",
    "numpy",
    "scipy",
    "toml",
    "qtlib"
]

dependencies.extend(extra_depends)

# Dynamic dependency configuration
setup(
    packages=['qt05_fvm_euler_1d'],
    package_dir={'': '.'},
    install_requires=dependencies
)
