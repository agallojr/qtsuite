
# README.md

Some workflow examples which involve the ORNL WCISCC code for HHL solvers. This is alo an example of using lwfm. We'll not try to exercise all aspects of these, just enough to show the basics.

It demonstrates how to handle use a project with dependencies declared in a manner which doesn't follow the current Python packaging standards (i.e. WCISCC). It also shows how to deal with a dependent project (like WCISCC) which uses older versions of important libraries like Qiskit.

So we'll pin ourselves here to their older dependencies, and show how we can construct circuits which then run on the latest version of Qiskit.

