#!/bin/bash
# Sweep runner for qp4p experiments
#
# Usage:
#   ./sweeper.sh <toml_file> <script>              # Run sweep
#   ./sweeper.sh <toml_file> <script> --dry-run    # Dry run
#   ./sweeper.sh <toml_file> <script> --group name # Run specific group
#
# Examples:
#   ./sweeper.sh mod0/input/hello_world_bell.toml mod0/src/hello_world_bell.py
#   ./sweeper.sh mod1/input/image_flip.toml mod1/src/image_flip.py --dry-run

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Use venv python if available
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# Run the sweeper with all arguments
$PYTHON helpers/src/qp4p_sweeper.py "$@"
