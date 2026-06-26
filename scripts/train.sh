#!/bin/bash
#OAR -n TabICL_Train
#OAR -l gpu=1,walltime=24:00:00


set -e

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Setup environment
source ~/thoth_storage/miniconda3/bin/activate ~/thoth_storage/miniconda3/envs/multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "Starting Training... (Job ID: $OAR_JOB_ID)"
python src/train.py "$@"
echo "Training complete."