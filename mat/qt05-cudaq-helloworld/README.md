
Running a docker container preloaded with CUDA Quantum:

Here we mount a local directory (~/proj) inside the container as /mac.
This was the latest CUDA-Q container at the time of writing.

docker run -it -v ~/proj:/mac nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.10.0

Then can use the VSCode Docker plugin to load source files which will have a good lib path taken from inside the container. The files themselves are on the local host.

The plugin can launch shells inside the container on which you can execute your app / scripts.

$ python helloworld.py

$ nvq++ helloworld.cpp -o helloworld.x
$ ./helloworld.x



