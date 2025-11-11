"""
help install dependencies

This setup.py ensures the fvm_euler_1d_solver repository is cloned and installed.
The git clone happens when this module is imported during the build process.
"""

import os
import subprocess
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build_py import build_py

# Clone the repo immediately when setup.py is executed
# This ensures it's available before uv tries to resolve dependencies

# Repository configuration
REPO_BRANCH = "feature/qiskit-hhl-2"
REPO_URL_BASE = "https://github.com/mhawwary/fvm_euler_1d_solver"
REPO_DIR = os.environ.get("FVM_REPO_DIR", "fvm_euler_1d_solver")


def clone_fvm_repo():
    """Clone the fvm euler 1d repository if it doesn't exist."""
    git_user_key = os.environ.get("GIT_USER_KEY")
    if not git_user_key:
        # You must set this environment variable for the installation to work
        raise ValueError("GIT_USER_KEY environment variable is not set - use 'user:key' format")
    # Construct authenticated URL from base URL
    repo_host = REPO_URL_BASE.replace("https://", "")
    REPO_URL = os.environ.get("FVM_REPO_URL",
        f"https://{git_user_key}@{repo_host}")
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {REPO_URL}...")
        subprocess.check_call(["git", "clone", "-b", REPO_BRANCH, REPO_URL])
        print(f"Successfully cloned {REPO_URL} to {REPO_DIR}")
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
        print(f"Found existing {REPO_DIR} directory - re-pulling")


def install_fvm_solver():
    """Ensure fvm_euler_1d_solver is installed with all dependencies.
    
    When fvm-euler-1d-solver is listed as a dependency in pyproject.toml,
    uv sync will automatically install it and its dependencies from setup.py
    (which reads from requirements.txt). This function ensures the repo is
    up to date before uv sync runs.
    """
    if os.path.exists(REPO_DIR):
        # Update the repo if it exists
        subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
        print(f"Updated {REPO_DIR} directory")
        print("fvm_euler_1d_solver will be installed via uv sync from pyproject.toml")


class PostInstallCommand(install):
    """Post-installation command that clones and installs fvm solver."""
    
    def run(self):
        clone_fvm_repo()
        install_fvm_solver()
        super().run()


class PostDevelopCommand(develop):
    """Post-development installation command that clones and installs fvm solver."""
    
    def run(self):
        clone_fvm_repo()
        install_fvm_solver()
        super().run()


class BuildPyCommand(build_py):
    """Custom build command that clones fvm solver before building."""
    
    def run(self):
        clone_fvm_repo()
        super().run()


# Clone immediately when setup.py is executed (before uv tries to resolve paths)
if __name__ != '__main__' and 'fvm_euler_1d_solver' not in sys.modules:
    try:
        clone_fvm_repo()
    except Exception as e:
        print(f"Warning: Could not clone fvm_euler_1d_solver: {e}")
        print("Make sure GIT_USER_KEY environment variable is set")


setup(
    packages=['qt05_fvm_euler_1d'],
    package_dir={'': '.'},
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
        'build_py': BuildPyCommand,
    },
)
