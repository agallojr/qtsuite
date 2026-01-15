#!/bin/bash
# Smoke test runner for qp4p experiment scripts
#
# Usage:
#   ./smoke_test.sh       # Run all tests (direct execution)
#   ./smoke_test.sh 0     # Run mod0 tests (direct execution)
#   ./smoke_test.sh 0s    # Run mod0 tests (via sweeper)
#   ./smoke_test.sh 1     # Run mod1 tests (direct execution)
#   ./smoke_test.sh 0 1   # Run mod0 and mod1 tests (direct)
#   ./smoke_test.sh 0s 1s # Run mod0 and mod1 tests (via sweeper)

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

# Pass all arguments to the Python script
$PYTHON helpers/src/smoke_test.py "$@"
