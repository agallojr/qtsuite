"""
setup project dependencies which don't use pyproject
run with:
uv run python config.py 
"""

import os
import subprocess

def install_requirements():
    """Clone the WCISCC2025 repo and install its requirements."""
    repo_url = "https://github.com/olcf/wciscc2025"
    repo_dir = "wciscc2025"
    requirements_file = os.path.join(repo_dir, "qlsa", "requirements.txt")

    # Clone the repository if it doesn't exist
    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url}...")
        subprocess.check_call(["git", "clone", repo_url])

    # Install requirements if the file exists
    if os.path.exists(requirements_file):
        print(f"Installing requirements from {requirements_file}...")
        subprocess.check_call(["uv", "pip", "install", "-r", requirements_file])
    else:
        print(f"Warning: Requirements file not found at {requirements_file}")


install_requirements()
