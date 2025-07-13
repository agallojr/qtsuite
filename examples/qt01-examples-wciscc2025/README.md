
This is a sub-project to drive the construction of quantum CFD test cases using the 
ORNL WCISCC code. They have their own dependencies including on Python and we'll be
running inside a Python virtual environment.

We assume you have uv and git installed. 
    - https://docs.astral.sh/uv/getting-started/installation/
    - https://github.com/git-guides/install-git

Python version dependency is defined in ./python-version

To setup:
- cd to this dir
- "uv venv" to make a new venv here
- "source ./.venv/bin/activate" to activate the virtual environment
- "uv sync --upgrade" to install dependencies

