# qtlib Usage Guide

## Quick Start

### 1. Add qtlib to your subproject

From your subproject directory (e.g., `mat/qt01-mat-sqd`):

```bash
# Option A: Using uv (recommended)
uv add --editable ../../qtlib

# Option B: Manual edit of pyproject.toml
# Add this line to your dependencies array:
# "qtlib @ file://../../qtlib",
# Then run: uv sync
```

### 2. Import and use

```python
from qtlib import get_cases_args

# Your code here
cases = get_cases_args()
```

### 3. Remove old local copy

If you had a local `get_wf_args.py`, you can now delete it and update imports:

```python
# Old:
from get_wf_args import get_cases_args

# New:
from qtlib import get_cases_args
```

## Example Migration

For a project at `mat/qt01-mat-sqd/`:

1. **Update pyproject.toml**:
   ```toml
   dependencies = [
       # ... existing dependencies
       "qtlib @ file://../../qtlib",
   ]
   ```

2. **Sync dependencies**:
   ```bash
   cd mat/qt01-mat-sqd
   uv sync
   ```

3. **Update imports in your Python files**:
   ```python
   from qtlib import get_cases_args
   ```

4. **Delete local copy** (if it exists):
   ```bash
   rm get_wf_args.py
   ```

## Directory Structure

```
qtsuite/
├── qtlib/                    # Shared library
│   ├── src/qtlib/
│   │   ├── __init__.py
│   │   └── workflow.py
│   ├── pyproject.toml
│   └── README.md
├── cfd/
│   └── qt02-cfd-qlsa-chal/  # Uses qtlib
├── mat/
│   └── qt01-mat-sqd/        # Can use qtlib
└── examples/
    └── ...                   # Can use qtlib
```

## Adding New Utilities to qtlib

1. Create your module in `qtlib/src/qtlib/your_module.py`
2. Export functions in `qtlib/src/qtlib/__init__.py`:
   ```python
   from .your_module import your_function
   __all__ = [..., "your_function"]
   ```
3. Document in `qtlib/README.md`
4. Keep dependencies minimal (add to qtlib's pyproject.toml only if necessary)
