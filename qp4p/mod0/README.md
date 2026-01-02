# Quantum Programming for Programmers (QP4P) - Module 0

## ▸ Terse Study Guide

There is little reason to duplicate here the enormous amount of information available online regarding quantum computing. Of the players in the emergent quantum computing field, two stand out for their contributions to the educational landscape: superconducting qubit vendor IBM makers of Qiskit the leading quantum programming framework, and to a lesser extent the photonics company Xanadu, makers of the Pennylane SDK. The educational materials from both companies are an excellent starting point - online documents, tutorials, videos on YouTube, and complete sequential courses.

As a basic introduction to utility, consider watching:
https://quantum.cloud.ibm.com/learning/en/courses/quantum-computing-in-practice/applications-of-qc

For the serious student, of particular note are the videos from John Watrous at IBM, especially the 18-part series "Understanding Quantum Computing", https://www.youtube.com/playlist?list=PLOFEBzvs-VvqKKMXX4vbi4EB1uaErFMSO

Individual modules for this course each include their own reading list and prerequisites. The Mike & Ike book has suggested chapter ranges for each module. Additional papers from the arxiv are suggested for each module's main topics.


## ▸ Goals of the Course

The title of this course is "Quantum Programming for Programmers". This course is designed to bridge the gap between classical programming knowledge and the unique paradigms of quantum computing, providing programmers with the foundational understanding needed to develop quantum applications. Its intended to be a bootstrapping, to allow students to learn the material more quickly in the company of other fellow and similar learners. This is not a math class, nor one in quantum physics or chemistry - the focus is on practical programming skills for practicing programmers, who themselves may or may not be scientific experts in another domain.

At the end of the course students will be able to:

- Explain the basics of quantum computing to peers, the value proposition (if any) to managers.
- Identify potential quantum utility and/or advantages and limitations for practical problems in their domain. Be able to discern the timing of that utility relative to vendor roadmaps.
- Use quantum programming frameworks to implement basic quantum algorithms and analyze results.
- Understand the impact of noise and error mitigation in quantum computations and the limitations of current hardware.
- Be able to form research collaborations internally and externally to further explore quantum applications.

The course is organized into four modules:

- Module 0: Introduction to Quantum Computing for Programmers         (60 min)
- Module 1: Quantum Programming Fundamentals, in the NISQ Era         (90 min)
- Module 2: Searching & Solving, w. CFD Use Cases                     (90 min)
- Module 3: Modeling Quantum Systems, w. MAT Use Cases                (90 min)

Module 0 is an overview and sales pitch. Module 1 gives hands-on experience with quantum programming concepts and the full stack, understanding the limitations of quantum and classical hardware and software, and handling the error inherent in the NISQ era. Armed with this by Module 2 you're ready to try and understand the perhaps eventual timing of applying quantum computing to CFD solver use cases pertinent to the business. Module 3 puts you on a course to make nearer-term contributions to the business and teams in areas of active research relating to modeling quantum systems.

There are prerequisites for the course, including basic programming knowledge and familiarity with linear algebra concepts. Each module has prerequisites. Some students may obtain credit for taking the course, which will be graded subjectively using code review and based on pass/fail. No work will be reviewed for credit 30 days after the end of the course.

Note there may not be a canonical solution to the homework problems. The homework will be intentionally speculative, potentially even at the edge of publishable, encouraging creative problem-solving and exploration of emerging quantum applications. Thus, the homework deliverable for each module is not code but rather a short publication. Students may collaborate in small teams for these portions. Documented failure is encouraged as part of the learning process.

There will be scheduled office hours, however, we hope, and have as a goal that students will quickly surpass the teacher.


## ▸ Module 0: Overview

How does quantum programming differ from classical? Are there applications today, if not, when?

Suitable for a general audience of programming practitioners, this overview module introduces the fundamental concepts of quantum computing including qubits, superposition, entanglement, quantum gates, circuits, and how these concepts differ from or are unique compared to classical programming. It covers the basic quantum computing workflow, performance limitations, and key algorithms in quantum computing for searching unsorted data, prime factorization, optimization,  solving linear systems of equations, and modeling quantum systems using quantum computing techniques and interim hybrid quantum-classical algorithms. Attendees will gain exposure to the roadmaps of leading quantum hardware and software providers and their projections for quantum utility or advantage in specific applications. A study guide for personal follow-up will be provided, and an optional office hour session will subsequently be made available.


## ▸ Module 0: Topics

+ Introduction and Overview
    - Executive context
    - Course overview and objectives: modules 1, 2, 3
    - What is quantum computing? Feynman's vision, probabilistic nature
    - Why study quantum computing? performance & scale beyond classical, Hilbert space, FeMOCO
    - tl;dr: safely ignore for now, abstractions will arrive before production utility
    - Utility & advantage: for what, when?
    - Current hardware landscape: hype & reality, metrics, types & connections, size, DARPA
    - Noise & error mitigation: code examples allow studying impact of noise
    - Current software landscape: maturity, interop, abstract, circuit, IDEs (code: hello_world)
    - IBM & NVIDIA, Python & C++, estimation

+ Quantum basics 
    - the mental shift, compared historical programming transitions
    - qubits vs. bits (code: state preparation & measurement, phase, magic, Bloch sphere)
    - superposition & Schrödinger's cat (code: Hadamard gate)
    - entanglement & the multi-body problem (code: Bell / GHZ states)
    - amplitude encoding (code: amplitude, image mirroring shots study)
    - phase kickback (code: phase_kickback)
    - observables

+ Quantum programming framework
    - Map the problem to a quantum-native and/or quantum-classical formulation
    - Optimize and transpile for the target hardware
    - Execute on quantum hardware or simulator
    - Analyze and post-process results
    - Mapping problems to known problems: complexity classes

+ Hybrid quantum-classical
    - Variational algorithms, pre- & post-processing, ansatz
    - HPC co-scheduling vs. cloud QPUs, new accelerator(s)
    - workflows & metadata

+ Overview of quantum algorithms
    - Shor's factoring: enterprise IT use cases
    - Grover's search (code: grovers)
    - Linear systems (code: ax=b): module 2 details CFD use cases
    - Variational Quantum Eigensolver (code: lattice): issues with NISQ variational algorithms
    - Quantum Phase Estimation (code: gs_qpe): module 3 details MAT use cases
    - other: QAOA, QML, TSP
    - formulating as a Hamiltonian
    - Algorithm Zoo

+ QP4P class: final plug
    - modules, labs, pre-reqs, schedule, SMEs, sign-up


## ▸ Module Library Dependencies

All examples for all modules use the same set of dependent libraries. These include Qiskit and its sub-libraries, numpy, matplotlib, and json for data processing and visualization. A suitable version of Python is also defined.

While the code examples use IBM's Qiskit, we do not require an account with the IBM cloud. All examples can run locally with simulated backends, although the set of backends is limited to those publicly available.

We recommend the use of uv (https://docs.astral.sh/uv/). We have published here a specification file (see ./pyproject.toml) that defines the required dependencies. 

With the libraries loaded you should be able to execute a simple test:
```bash
python mod0/src/hello_world.py
```


## ▸ Running the Sweeps, Gathering Data & Metadata

In the later modules, we show examples of building and executing quantum circuits in more complex algorithmic workflows. The core of these implementations take many command line parameters and return structured data (JSON). Thus workflows can be defined in an input file (TOML format) and swept using the provided qp4p_sweeper tool, with any interactive displays saved for the end of the workflow and later analysis. The core example scripts include a "--no-display" switch to suppress display output which can be useful in workflows.

The sweeper tool is located in the qp4p package and can be invoked with various configuration options to run parameter sweeps. One thing it does is keep the output isolated in its own directory for easy organization. Each case in the sweep is also stored in its own separate directory. By separating data generation from post-processing we allow for multiple post-processing steps.

The toml file allows groupings of swept cases and will lay down metadata about the group which can be used by postprocessing scripts to identify the cases within the group and other parameters for the post-processing step. Variables in the toml with a "_" prefix are arguments to the sweeper itself and are not swept (e.g. path to root outdir).

For example, here we run the core image_flip demo with a parameter sweep defined in the toml input file, running just one group of cases defined in that input file:
```bash
python -m qp4p_sweeper mod1/src/image_flip.py mod1/input/image_flip.toml --group size_study
```

The core algorithm json output includes many metadata fields about the problem, solution, and execution environment. Students are encouraged to consider adding their own custom metadata fields to capture domain-specific insights. These can include performance metrics, hardware-specific details, or any other relevant information for their particular use case. Such metadata can be invaluable for tracking experimental results and comparing different approaches. 

Quantum computing by its nature at this time can create many pieces of interim artifacts and results, and they have importance in the current research. Keeping track of these artifacts is essential for comprehensive analysis and reproducibility. Students are encouraged to consider what workflow tools work best for them in this regard.

Additionally, quantum solutions at this time are often hybrid classical-quantum approaches where the classical part can itself be heavyweight and a bottleneck. HPC can be useful here. The student is encouraged to consider which workflow tools work best for them in managing these hybrid workloads, such as containerization, with on-prem and cloud orchestration.


## ▸ Some Illustrative Examples

Prepare a custom qubit state with a global noise model:
```bash
python mod0/src/qubits.py prep -n 5 --state 1 --t1 50 --t2 30
```

Generate a GHZ state on a large backend. This will dump out copious information about the current backend configuration - on a real quantum computer this can change often / hourly. Note how transpiled circuit size differs from the initial abstract representation. Notice the global t1 and t2 error rates are not set but that each individual qubit and qubit pair has a noise metric. The coupling map shows not all qubits are connected, which affects circuit compilation. Notice the practical limits of the display for large numbers of qubits.
```bash
python mod0/src/qubits.py ghz --backend brooklyn -n 10
```

Suppress pop-ups:
```bash
python mod0/src/qubits.py ghz --backend brooklyn -n 10 --no-display
```

Solve an Ax=b problem using VQLS (Variational Quantum Linear Solver) on an idealized hardware. A and b are pseudo-random and A is symmetric positive definite. Module 2 will cover solving this problem with applications to CFD and other scientific computing problems in more depth.
```bash
python mod1/src/ax_equals_b_vlqs.py --size 8 --maxiter 500
```

Run the image flip experimenting with different shot counts. At the end of the run, see ~/qp4p/ for the results directory. An image viewer will popup allowing you to step through the results and see the impact of increasing shots.
```bash
python -m qp4p_sweeper mod1/src/image_flip.py mod1/input/image_flip.toml --group shots_study
```

Computing the ground state energy of H2 with Quantum Phase Estimation (QPE) using variational methods. The sweep increases the number of ancilla qubits in the phase estimation circuit, increasing precision. Notice how the variational approach can overshoot the exact result as precision increases. In Module 3 we will consider use cases in quantum chemistry and material science.
```bash
python -m qp4p_sweeper mod1/src/gs_qpe.py mod1/input/gs_qpe.toml --group h2_ancilla
```

## ▸ Supported Backends

At the time of this writing, the code supports the following backends with the "--backend" flag:

- **1 qubit**: armonk
- **5 qubits**: athens, belem, bogota, casablanca, essex, lima, london, manila, ourense, quito, rome, santiago, valencia, vigo, yorktown
- **7 qubits**: jakarta, lagos, nairobi
- **14 qubits**: melbourne
- **15 qubits**: guadalupe
- **16 qubits**: almaden, singapore
- **20 qubits**: boeblingen, johannesburg, poughkeepsie
- **27 qubits**: cairo, cambridge, hanoi, kolkata, montreal, mumbai, paris, sydney, toronto
- **53 qubits**: rochester
- **65 qubits**: brooklyn, manhattan, washington


## ▸ Module 0 Post-Work

If you intend to take module 1, we suggest reviewing these questions to better prepare:

- Run the samples provided using the sweeper. Are you able to construct your own parameterized studies, use your own post-process analysis tools? Run on different backends and compare results? Customize the backend with coupling maps and noise models?

- How large a GHZ state can you practically construct on your machine? Without and with noise?

- At the edge of the GHZ state you can simulate, after multiple runs, what can be said about the reliability and consistency of the results? As a function of qubits, t1, t2?

- How large an image can you mirror, to what fidelity? With and without noise in the sim? What is the impediment? How might we leverage classical HPC in this problem? What's the impact of shots?

- Run a study of the lattice model, to understand the impact of size, ansatz, noise, optimizer, and iterations. Experiment with models of different legacy IBM machines (backends) and compare performance.

- See also the documentation for Module 1: mod1/README.md for more pre-requisites.

Module 0 post-work will not be graded, but is highly recommended for deeper understanding.

