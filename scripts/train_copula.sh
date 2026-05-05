#!/bin/bash
#OAR -n AttentionalCopula_Train
#OAR -l gpu=1,walltime=24:00:00
#OAR -t besteffort
#OAR -O train_copula_output.%jobid%.txt
#OAR -E train_copula_error.%jobid%.txt

# 1. Navigate to project root relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# 2. Setup environment
source ~/thoth_storage/miniconda3/bin/activate ~/thoth_storage/miniconda3/envs/multivariate-icl

# 3. Prevent path pollution from .local
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 4. Launch training
python src/copula_train.py "$@"
