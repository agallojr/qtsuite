#!/usr/bin/env python3
"""
Sample postprocessing script that echoes its arguments.

This script is called by the sweeper after all cases in a group complete.
It receives a single JSON file path containing the postproc context.

Usage:
    python postproc_echo.py /path/to/_postproc_groupname.json
"""

import sys
import json
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python postproc_echo.py <postproc_json>")
        sys.exit(1)
    
    # Load postproc context from JSON file
    postproc_json = Path(sys.argv[1])
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    print("=" * 60)
    print("POSTPROC ECHO")
    print("=" * 60)
    print(f"Group ID: {context.get('group_id', 'N/A')}")
    print(f"Run ID: {context.get('run_id', 'N/A')}")
    print(f"Run Dir: {context.get('run_dir', 'N/A')}")
    print(f"Number of case directories: {len(context.get('case_dirs', []))}")
    print()
    
    for i, case_dir_str in enumerate(context.get("case_dirs", []), 1):
        case_dir = Path(case_dir_str)
        print(f"Case {i}: {case_dir.name}")
        
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
