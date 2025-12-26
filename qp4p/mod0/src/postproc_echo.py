#!/usr/bin/env python3
"""
Sample postprocessing script that echoes its arguments.

This script is called by the sweeper after all cases in a group complete.
It receives the list of case directories as command-line arguments.

Usage:
    python postproc_echo.py /path/to/case1 /path/to/case2 ...
"""

import sys
import json
from pathlib import Path


def main():
    print("=" * 60)
    print("POSTPROC ECHO")
    print("=" * 60)
    print(f"Number of case directories: {len(sys.argv) - 1}")
    print()
    
    for i, arg in enumerate(sys.argv[1:], 1):
        case_dir = Path(arg)
        print(f"Case {i}: {case_dir}")
        
        # Try to read params.json if it exists
        params_file = case_dir / "params.json"
        if params_file.exists():
            with open(params_file, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"  Params: {json.dumps(params, indent=4)}")
        
        # Try to read stdout.json if it exists
        stdout_file = case_dir / "stdout.json"
        if stdout_file.exists():
            with open(stdout_file, "r", encoding="utf-8") as f:
                try:
                    output = json.load(f)
                    # Just show a summary
                    if "analysis" in output:
                        print(f"  Analysis: {output['analysis']}")
                except json.JSONDecodeError:
                    print("  (stdout.json is not valid JSON)")
        
        print()
    
    print("=" * 60)
    print("POSTPROC ECHO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
