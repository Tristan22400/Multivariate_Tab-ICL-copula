#!/bin/bash
#OAR -n TabICL_Generate
#OAR -l gpu=1,walltime=02:00:00


set -e

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Setup environment
source ~/thoth_storage/miniconda3/bin/activate ~/thoth_storage/miniconda3/envs/multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting PIT generation... (Job ID: $OAR_JOB_ID)"
python src/generate_pit_dataset.py "$@"
echo "Generation complete."