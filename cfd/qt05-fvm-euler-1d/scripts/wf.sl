#!/bin/bash
#SBATCH -A ARD189
#SBATCH -J fvm_euler_1d
#SBATCH -o /ccs/proj/ard189/lwfm/out/slurm
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1

srun /ccs/proj/ard189/lwfm/src/fvm_euler_1d_solver/wf.sh
