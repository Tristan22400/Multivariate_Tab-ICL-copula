#!/bin/bash
#OAR -n TabICL_Generate
#OAR -l gpu=1,walltime=16:00:00


set -e

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Setup environment — use env Python directly to avoid conda init requirements
PYTHON=~/thoth_storage/miniconda3/envs/multivariate-icl/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting PIT generation... (Job ID: $OAR_JOB_ID)"
$PYTHON src/generate_pit_dataset.py "$@"
echo "Generation complete."