#!/bin/bash
# Clone fvm_euler_1d_solver and prepare for uv sync
# Run this before 'uv sync' to set up dependencies
#
# Usage:
#   export GIT_USER_KEY="username:token"
#   source .gitconfig-helper.sh

set -e

REPO_BRANCH="feature/qiskit-hhl-2"
REPO_URL_BASE="https://github.com/mhawwary/fvm_euler_1d_solver"
REPO_DIR="fvm_euler_1d_solver"

if [ -z "$GIT_USER_KEY" ]; then
    echo "Error: GIT_USER_KEY environment variable not set"
    echo "Set it using: export GIT_USER_KEY='username:token'"
    return 1 2>/dev/null || exit 1
fi

# Construct authenticated URL
REPO_HOST="${REPO_URL_BASE#https://}"
REPO_URL="https://${GIT_USER_KEY}@${REPO_HOST}"

# Clone or update the repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning $REPO_URL_BASE (branch: $REPO_BRANCH)..."
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    echo "Successfully cloned to $REPO_DIR"
else
    echo "Found existing $REPO_DIR directory"
    echo "Updating repository..."
    git -C "$REPO_DIR" pull
    echo "Repository updated"
fi

echo ""
echo "Dependencies ready. You can now run: uv sync"
