# qtlib

Shared utilities for qtsuite projects.

## Installation

From any qtsuite subproject, add qtlib as an editable dependency:

```bash
cd /path/to/your/subproject  # e.g., cfd/qt02-cfd-qlsa-chal
uv add --editable ../../qtlib
```

Or manually add to your `pyproject.toml`:

```toml
dependencies = [
    # ... other dependencies
    "qtlib @ file://../../qtlib",
]
```

Then run:
```bash
uv sync
```

## Usage

### Workflow utilities

```python
from qtlib import get_cases_args

# Parse TOML workflow file from command line
cases = get_cases_args()
```

## Adding new utilities

1. Add your module to `src/qtlib/`
2. Export public functions in `src/qtlib/__init__.py`
3. Keep dependencies minimal (currently none)
4. Document usage in this README
