
qt05-examples-benchqc

This directory contains examples for benchmarking quantum chemistry computations using Qiskit and Jarvis libraries. The scripts are designed to set up quantum circuits for simulating the electronic structure of molecules, allowing for various configurations and optimizations.

The code uses very old libraries (see pyproject.toml), and no effort has been made to update. We have however suppressed the warnings, else you'd not see the results through all the warning noise.

This is a very basic example which can be extended as the use cases warrant. For example, we could repackage the code to use lwfm to leverage the IBM site driver in a virtual environment to permit use of the latest backends and drivers. There are other examples in this git repo which show how to do that, and we can again as use cases warrant.



