#!/bin/bash

# Set LWFM_HOME if not already set
: ${LWFM_HOME:=/ccs/proj/ard189/lwfm}
export LWFM_HOME
echo $LWFM_HOME

proj_dir=$LWFM_HOME

cd $proj_dir/src/fvm_euler_1d_solver

. .venv/bin/activate

python wf.py input/00-in.toml

../lwfm/lwfm.sh stop



