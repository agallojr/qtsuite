#!/bin/bash
#SBATCH -A ARD189
#SBATCH -J fvm_euler_1d
#SBATCH -o /ccs/proj/ard189/agallojr/out/fvm-euler-1d-solver/slurm-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1

# Usage: sbatch wf.sl <input_toml> [--resume-workflow] [--submit-next]

srun /ccs/proj/ard189/agallojr/src/qtsuite/cfd/qt05-fvm-euler-1d/scripts/wf.sh "$@"
