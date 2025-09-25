
Source Code
=====

ORNL provided code used in their Frontier training (https://github.com/olcf/hands-on-with-frontier). The pertinent HHL portion originated with the Winter Challenge 2025 (https://github.com/olcf/wciscc2025), and notably the key `linear_solvers` library used in the current code comes from (https://github.com/jw676/quantum_linear_solvers) which is a fork of our work (https://github.com/agallojr/quantum_linear_solvers) which we of course forked from them and updated it to work with Qiskit 2.0+, from its 0.x origins. We then forked hands-on-with-frontier - the subfolder which contains the QLSA code.

In order to orchestrate workflows and keep artifacts separated, we use some workflow tooling we developed (https://github.com/lwfm-proj/lwfm) which is inspired by our industrial experience (https://link.springer.com/chapter/10.1007/978-3-031-23606-8_16).

The ORNL code contains references to IonQ and IQM drivers. In the case of IQM, these are pinned to older version of Qiskit. We have a strong preference to stay on the tip of the major lib version trees like Qiskit, thus in our fork of the ORNL code we removed these dependencies. 


