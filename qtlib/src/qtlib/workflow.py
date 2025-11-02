"""
Get workflow arguments from a TOML file
"""

#pylint: disable=invalid-name, missing-function-docstring, global-statement
#pylint: disable=broad-exception-caught, logging-fstring-interpolation

from pathlib import Path
import sys
from itertools import product
from datetime import datetime
import time

from logging import getLogger

logger = getLogger(__name__)

try:
    import tomllib  # Python 3.11+
except (ModuleNotFoundError, ImportError):
    try:
        import tomli as tomllib  # Python 3.7-3.10
    except ImportError:
        tomllib = None  # Will be handled below


def log_with_time(message, time_array):
    current_time = time.time()
    workflow_start = time_array[0]
    delta = current_time - workflow_start
    cumulative = current_time - workflow_start
    timestamp = datetime.now().strftime("%H:%M:%S")

    if len(time_array) > 1 and time_array[1] is not None:
        case_start = time_array[1]
        case_cumulative = current_time - case_start
        logger.info(
            f"[{timestamp}] [T+{cumulative:.2f}s] "
            f"[C+{case_cumulative:.2f}s] [Δ{delta:.2f}s] {message}"
        )
    else:
        logger.info(
            f"[{timestamp}] [T+{cumulative:.2f}s] "
            f"[Δ{delta:.2f}s] {message}"
        )


def expand_case_lists(case_id, case_params):
    """
    Expand a case with list-valued parameters into multiple subcases.
    
    For example, if a case has:
        qc_shots = [100, 1000, 10000]
        NQ_MATRIX = 2
    
    This will expand into 3 subcases:
        case_id_0: qc_shots=100, NQ_MATRIX=2
        case_id_1: qc_shots=1000, NQ_MATRIX=2
        case_id_2: qc_shots=10000, NQ_MATRIX=2
    
    Parameters
    ----------
    case_id : str
        The original case ID
    case_params : dict
        The case parameters (may contain lists)
    
    Returns
    -------
    list of tuples
        List of (expanded_case_id, expanded_params, metadata) tuples
        where metadata contains info about which params were lists
    """
    # Find which parameters are lists
    list_params = {}
    scalar_params = {}

    for key, value in case_params.items():
        if isinstance(value, list):
            list_params[key] = value
        else:
            scalar_params[key] = value

    # If no lists, return the original case with empty metadata
    if not list_params:
        metadata = {
            "_original_case_id": case_id,
            "_list_params": [],
            "_scalar_params": list(scalar_params.keys())
        }
        params_with_meta = case_params.copy()
        params_with_meta["_metadata"] = metadata
        return [(case_id, params_with_meta)]

    # Generate all combinations of list values
    param_names = list(list_params.keys())
    param_values = [list_params[name] for name in param_names]

    expanded_cases = []
    for i, combination in enumerate(product(*param_values)):
        # Create new case ID
        expanded_id = f"{case_id}_{i}"

        # Create new params dict with this combination
        expanded_params = scalar_params.copy()
        for param_name, param_value in zip(param_names, combination):
            expanded_params[param_name] = param_value

        # Add metadata about the expansion
        metadata = {
            "_original_case_id": case_id,
            "_list_params": param_names,  # Which params were lists
            "_scalar_params": list(scalar_params.keys()),
            "_combination_index": i
        }
        expanded_params["_metadata"] = metadata

        expanded_cases.append((expanded_id, expanded_params))

    return expanded_cases


def get_cases_args(workflow_toml):
    """Get workflow arguments from a TOML file"""

    wf_toml_path = Path(workflow_toml)
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
            with open(wf_toml_path, "r", encoding="utf-8") as f:
                data = toml.load(f)
    except Exception as e:
        print(f"Error decoding TOML: {e}")
        sys.exit(1)

    # build a dict with all cases (including "global")
    # First, separate global from cases
    global_params = data.get("global", {})
    cases_dict = {"global": global_params}

    # Expand each case that has list-valued parameters
    for key in data:
        if key == "global":
            continue

        # Expand this case (returns list of tuples)
        expanded = expand_case_lists(key, data[key])

        # Add all expanded subcases to the dict, merging in global params
        for expanded_id, expanded_params in expanded:
            # Start with global params, then overlay case-specific params
            merged_params = global_params.copy()
            merged_params.update(expanded_params)
            cases_dict[expanded_id] = merged_params

    return cases_dict
