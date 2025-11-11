#!/bin/bash

# Usage: wf.sh <input_toml> [--resume-workflow] [--submit-next]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_toml> [--resume-workflow] [--submit-next]"
    exit 1
fi

# Set LWFM_HOME if not already set
: ${LWFM_HOME:=/ccs/proj/ard189/agallojr}
export LWFM_HOME
echo $LWFM_HOME

proj_dir=$LWFM_HOME

cd $proj_dir/src/qtsuite/cfd/qt05-fvm-euler-1d

. .venv/bin/activate

# Pass all arguments to wf.py
python wf.py "$@"




