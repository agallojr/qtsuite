"""
Get workflow arguments from a TOML file
"""

#pylint: disable=invalid-name

import argparse
from pathlib import Path
import sys

try:
    import tomllib  # Python 3.11+
except (ModuleNotFoundError, ImportError):
    try:
        import tomli as tomllib  # Python 3.7-3.10
    except ImportError:
        tomllib = None  # Will be handled below

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
        if tomllib is not None:
            # Use tomllib/tomli (binary mode)
            with open(wf_toml_path, "rb") as f:
                data = tomllib.load(f)
        else:
            # Fallback to toml package for very old environments
            import toml
            with open(wf_toml_path, "r") as f:
                data = toml.load(f)
    except Exception as e:
        print(f"Error decoding TOML: {e}")
        sys.exit(1)

    # build a dict with all cases (including "global")
    cases_dict = {}
    for key in data:
        cases_dict[key] = data[key]

    return cases_dict
