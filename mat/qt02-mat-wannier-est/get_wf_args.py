"""
Get workflow arguments from a TOML file
"""

#pylint: disable=invalid-name

import argparse
from pathlib import Path
import sys
import toml

def get_cases_args():
    """Get workflow arguments from a TOML file"""

    # parse CLI args for workflow input file (required)
    parser = argparse.ArgumentParser(description="Run HHL workflow cases from a TOML definition")
    parser.add_argument("workflow_toml", metavar="WORKFLOW_TOML", help="Path to input TOML file")
    args = parser.parse_args()

    wf_toml_path = Path(args.workflow_toml)
    if not wf_toml_path.is_file():
        print(f"Error: {wf_toml_path} not found.")
        sys.exit(1)

    # load the TOML file of test case inputs
    try:
        with open(wf_toml_path, "r") as f:
            data = toml.load(f)
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML: {e}")
        sys.exit(1)

    # build a dict with all cases (including "global")
    cases_dict = {}
    for key in data:
        cases_dict[key] = data[key]

    return cases_dict
