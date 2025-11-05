#!/bin/bash

# Set LWFM_HOME if not already set
: ${LWFM_HOME:=/ccs/proj/ard189/agallojr}
export LWFM_HOME
echo $LWFM_HOME

proj_dir=$LWFM_HOME

cd $proj_dir/src/qtsuite/cfd/qt05-fvm-euler-1d

. .venv/bin/activate

python wf.py input/01-in.toml




