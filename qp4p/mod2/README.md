# Quantum Programming for Programmers (QP4P) - Module 2

## ▸ Module 2: Prerequisites

- Attend QP4P Module 1, with its prerequisites and homework.

- Read "Quantum Computation and Quantum Information" by Mike & Ike, Chapters 5-6, 9.

- Read "Quantum algorithm for solving linear systems of equations", Harrow, Hassidim, Lloyd, https://arxiv.org/abs/0811.3171

- Read "Variational Quantum Linear Solver", Bravo-Prieto et al, https://arxiv.org/abs/1909.05820

- Read "Quantum CFD in 2025: An Industrial Perspective with Use Cases", Gallo & Alhawwary, distributed in this repository.


## ▸ Module 2: Overview

Module 2: Searching & Solving, w. CFD Use Cases 

How can quantum be used for search, optimization, and solving linear systems as in CFD?

Gain exposure to key algorithms in quantum computing: for searching unsorted data, optimization, and solving linear systems of equations, and understand their reported advantage relative to classical. Understand the current state of research in applying quantum algorithms to real-world problems in fluid flow with a report-out from a subject matter expert. Run examples of basic algorithms against toy CFD cases on a variety of noisy backends. Produce a new benchmark towards tracking industrial utility.


## ▸ Module 2: Topics

+ Q & A

+ Grover's search (code: grovers)
    - algorithm overview and implementation
    - applications
    - quantum subroutines & accelerators in a hybrid environment

+ Variational Algorithms & Optimization
    - QAOA overview and implementation
    - VQLS overview and implementation
    - as a NISQ era stop-gap, & issues with 

+ HHL [SME]
    - algorithm overview
    - Hele-Shaw case
    - 1D supersonic divergent nozzle

+ <5 min break>

+ Current CFD Research & Future Work [SME]

+ Homework 2

+ Q & A

+ Office Hours


## ▸ Module 2: Homework

Read "Quantum CFD: Utility Benchmarking" by Gallo & Alhawwary et al., distributed in this repository. The proposal describes benchmarking approaches for quantum computing with application to CFD, studying a cube of configurations: code x case x backend. For example "HHL for 1D supersonic divergent nozzle on IBM Heron-class backend". Consider where you can, practically speaking, add results to the cube. Make a plan, consult the CFD SME, implement it, and document your findings in a short report (2-3 pages at most). If your results are used in our study, you may be listed as co-author on any resulting publication.

You may if desired work in small teams.

Consider that industrial quantum utility is not likely to be found for your chosen case at this time in the evolution of quantum computing. Focus instead on demonstrating feasibility, identifying bottlenecks, and establishing a foundation for future utility. Consider therefore how to encode your workflow(s) to maximize their potential for re-run and reuse. Constructing fictitious backends is also permitted (e.g. an all-to-all machine with constant error rates on all qubits and connections).

[Backup homework: Not a CFD person and find the above daunting? Then try the backup homework instead - implement the Oak Ridge "Winter Challenge 2025" - a simplified exercise to evaluate the HHL algorithm on the Hele-Shaw flow problem. See here: https://github.com/olcf/wciscc2025]

