#!/bin/bash
# Helper script to configure git to use GIT_USER_KEY for authentication
# Run this before 'uv sync' if the fvm_euler_1d_solver repo is private
#
# Usage:
#   export GIT_USER_KEY="username:token"
#   source .gitconfig-helper.sh
#   uv sync

if [ -z "$GIT_USER_KEY" ]; then
    echo "Error: GIT_USER_KEY environment variable not set"
    echo "Set it using: export GIT_USER_KEY='username:token'"
    exit 1
fi

# Configure git to use the credentials for this session
git config --global url."https://${GIT_USER_KEY}@github.com/".insteadOf "https://github.com/"

echo "Git configured to use GIT_USER_KEY for GitHub authentication"
echo "You can now run: uv sync"
