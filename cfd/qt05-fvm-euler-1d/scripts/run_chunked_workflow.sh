#!/bin/bash
# Script to run chunked workflow iterations until completion
# Usage: ./scripts/run_chunked_workflow.sh input/03-chunked-in.toml

set -e  # Exit on error

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_toml_file>"
    echo "Example: $0 input/03-chunked-in.toml"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get savedir from TOML (default to ~/.lwfm/out/fvm-euler-1d-solver)
SAVEDIR=$(grep -E "^\s*savedir\s*=" "$INPUT_FILE" | head -1 | sed 's/.*=\s*"\([^"]*\)".*/\1/' | sed "s|~|$HOME|" | tr -d ' ')
if [ -z "$SAVEDIR" ]; then
    SAVEDIR="$HOME/.lwfm/out/fvm-euler-1d-solver"
fi

# Ensure savedir exists
mkdir -p "$SAVEDIR"

echo "Starting chunked workflow with input: $INPUT_FILE"
echo "Output directory: $SAVEDIR"
echo "=========================================="

# First iteration - create new workflow
echo ""
echo "Running iteration 0 (new workflow)..."
python wf.py "$INPUT_FILE" 2>&1 | tee "$SAVEDIR/wf_iter0.log"

# Check if we need to continue
ITERATION=1
while true; do
    # Check for completion messages in the log
    if grep -q "WORKFLOW COMPLETE" "$SAVEDIR/wf_iter$((ITERATION-1)).log" 2>/dev/null; then
        echo ""
        echo "=========================================="
        echo "Workflow completed!"
        if grep -q "SOLUTION CONVERGED" "$SAVEDIR/wf_iter$((ITERATION-1)).log" 2>/dev/null; then
            echo "Status: Solution converged"
        elif grep -q "MAX ITERATIONS REACHED" "$SAVEDIR/wf_iter$((ITERATION-1)).log" 2>/dev/null; then
            echo "Status: Maximum iterations reached"
        fi
        echo "=========================================="
        break
    fi
    
    # Check if next iteration command is present
    if grep -q "NEXT ITERATION COMMAND:" "$SAVEDIR/wf_iter$((ITERATION-1)).log" 2>/dev/null; then
        echo ""
        echo "Running iteration $ITERATION (resuming workflow)..."
        python wf.py --resume-workflow "$INPUT_FILE" 2>&1 | tee "$SAVEDIR/wf_iter$ITERATION.log"
        ITERATION=$((ITERATION + 1))
    else
        echo ""
        echo "Warning: Unexpected output from wf.py"
        echo "Last output saved in $SAVEDIR/wf_iter$((ITERATION-1)).log"
        exit 1
    fi
done

echo ""
echo "Chunked workflow script completed."
echo "All logs saved in: $SAVEDIR"
