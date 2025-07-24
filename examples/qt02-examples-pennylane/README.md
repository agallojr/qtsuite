
# README.md
# PennyLane Example with lwfm

An example of using PennyLane to create a quantum circuit and running it on an IBM 
backend using lwfm.

We'll not show all the lwfm bells and whistles here - see the other examples.

1. Make a simple circuit, PennyLane style.
2. Execute it on PennyLane's default simulator device right here.
3. Prepare to toss to the IBM backend by converting it to industry "standard" QASM.
4. Make a workflow context, define the job, and submit it to the IBM backend.
5. Wait for the job to complete and compare the results to the above.

This shows that different backends might produce results in different formats, and the 
app (i.e. this script / workflow) would be responsible for that kinds of handling. lwfm 
normalizes what it can.

This also shows you can avoid using the lwfm "Workflow" stuff if you want, and just
get the benefit of the backend interop.

