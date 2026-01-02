# Quantum Programming for Programmers (QP4P) - Module 2

## ▸ Module 2: Prerequisites

- Attend QP4P Module 1, with its prerequisites and homework.

- Read "Quantum Computation and Quantum Information" by Mike & Ike, Chapters 5-6, 9.

- Read "Quantum algorithm for solving linear systems of equations", Harrow, Hassidim, Lloyd, https://arxiv.org/abs/0811.3171

- Read "Variational Quantum Linear Solver", Bravo-Prieto et al, https://arxiv.org/abs/1909.05820

- Read "Quantum CFD in 2025: An Industrial Perspective with Use Cases", Gallo & Alhawwary, distributed in this repository.


## ▸ Module 2: Overview



## ▸ Module 2: Topics


- variational algorithms & optimization
- HHL and VQLS algorithms - HS & 1D nozzle flow

Module 2: Searching and Solving
How can quantum be used for search, optimization, and solving linear systems as in CFD?
2 hours lecture & lab in-person / virtual, 1-2 hours post-work

Gain exposure to key algorithms in quantum computing: for searching unsorted data, prime factorization, optimization, and solving linear systems of equations, and understand their reported advantage relative to classical. Understand the current state of research in applying quantum algorithms to real-world problems in fluid flow with a report-out from a subject matter expert. Run a simple CFD problem in class using the HHL algorithm on a quantum simulator with and without a noise model. Continue your study by exercising the VQLS algorithm for solving the same system in a hybrid quantum-classical setting and attend office hours for support.




+ Module 2: Searching and Solving
    + Key takeaways / value proposition
        - Gain exposure to key algorithms in quantum computing: for solving systems of equations, searching unsorted data, prime factorization, optimization.
        - Understand the current state of research in applying quantum algorithms to real-world problems in fluid dynamics.
        - Run a simple CFD problem using the HHL algorithm on a quantum simulator with and without a noise model.
        - Continue your study by exercising VQLS algorithm for solving the same system.
    + Pre-Work
        - Read up on HHL, VQLS, QAOA, Shor's factorization and Grover's Search algorithm.
    + Lecture
        + Quantum algorithms overview
            - Grover's Search
            - Shor's Factorization
            - Quantum Approximate Optimization Algorithm (QAOA)
            - HHL Algorithm for Linear Systems
            - Variational Quantum Linear Solver (VQLS)
    + Lab
    + Post-Work




## ▸ Module 2: Homework

Read "Quantum CFD: Utility Benchmarking" by Gallo & Alhawwary et al., distributed in this repository. The proposal describes benchmarking approaches for quantum computing with application to CFD, studying a cube of configurations: code x case x backend. For example "HHL for 1D supersonic divergent nozzle on IBM Heron-class backend". Consider where you can, practically speaking, add results to the cube. Make a plan, consult the CFD SME, implement it, and document your findings in a short report (2-3 pages at most). If your results are used in our study, you may be listed as co-author on any resulting publication.

You may work in small teams.

Consider that industrial quantum utility is not likely to be found for your chosen case at this time in the evolution of quantum computing. Focus instead on demonstrating feasibility, identifying bottlenecks, and establishing a foundation for future utility. Consider therefore how to encode your workflow(s) to maximize their potential for re-run and reuse.

