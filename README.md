
qtsuite - quantum test suite(s)

A project of projects. 

## Introduction

Each project is a suite of tests for a specific quantum computing framework or library or purpose.

Each project contains its own dependencies, defined in a pyproject.toml file. This might also include a specific version of Python, which will get installed when following the instructions below.

We recommend the use of uv for managing virtual environments.

After construction of the virtual environment, one or more test cases contained in the project can be executed.


## Installation

1. Get and install uv for your platform. See: https://docs.astral.sh/uv/
2. cd to a project directory and make a virtual environment for it with uv:
    $ uv venv --native-tls
    $ . venv/bin/activate
    $ uv sync --upgrade --native-tls  # the -native-tls is useful for corporate networks
3. Run the tests using instructions for that project.


## Naming convention

Each project is named with a prefix of `qtsuite-` followed by the specific framework or purpose, e.g., `qtsuite-qiskit`, `qtsuite-pennylane`, etc. For example, `qt01-cfd-wciscc2025` is the 1st test suite in the set pertaining to CFD, and it happens to pertain to the `wciscc2025` tooling. We suggest the suite numbering be immutable and to use a "-DEPRECATED" suffix for any deprecated test suite. Within the `qt01-cfd-wciscc2025` suite, the test cases are named with a prefix of `qt01-`, e.g., `test_01`, `test_02`, etc. This allows for easy identification of the suite and test case, and again, we recommend the test cases be immutable and marked as deprecated if they are no longer relevant.


## Use of lwfm

Some but perhaps not all the suite sub-projects use the lwfm (Lightweight Workflow Manager) for managing workflows. If a project uses lwfm, the lwfm service is expected to be running. A ~/.lwfm/ should contain a sites.toml which defines the compute sites - security credentials, etc. to be used for that site. 

See lwfm for instructions. We recommend using the development branch since this is active "research grade" software under active development. https://github.com/lwfm-proj/lwfm/tree/develop


