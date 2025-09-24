
Source Code
=====

ORNL provided code used in their Frontier training (https://github.com/olcf/hands-on-with-frontier). The pertinent HHL portion originated with the Winter Challenge 2025 (https://github.com/olcf/wciscc2025), and notably the key `linear_solvers` library used in the current code comes from (https://github.com/jw676/quantum_linear_solvers) which is a fork of our work (https://github.com/agallojr/quantum_linear_solvers) whereby we updated it to work with Qiskit 2.0+ from its 0.x origins. 

The moving target which is Qiskit and these early stage quantum tools is a notable impediment to reproducibility going forward. IBM and others will openly state this is research-grade software and that it is subject to backward-incompatible breaking changes. Reproducibility is a problem with computational science in general, and is very evident with quantum computing. 

In order to orchestrate workflows and keep artifacts separated, we use some workflow tooling we developed (https://github.com/lwfm-proj/lwfm) which is inspired by our industrial experience (https://link.springer.com/chapter/10.1007/978-3-031-23606-8_16).

The ORNL code contains references to IonQ and IQM drivers. In the case of IQM, these are pinned to older version of Qiskit. We have a strong preference to stay on the tip of the major lib version trees, thus we forked the ORNL code to remove these dependencies. 



