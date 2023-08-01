#!/bin/bash
#SBATCH --job-name=30
#SBATCH --partition=math-alderaan-gpu
#SBATCH --nodelist=math-alderaan-h01
#SBATCH --time=7-00:00:00                  # Max wall-clock time 1 day 1 hour
#SBATCH --gres=gpu:a100:1  

# run tensorflow in singularity container
# redirect output to a file so that it can be inspected before the end of the job
export HDF5_USE_FILE_LOCKING=FALSE
singularity exec /home/davoudis/tensorflow_nvidia.sif python3 bin/pretrain_proteinbert.py --dataset-file=/home/ceas/davoudis/paper/data/cluster-30/pretrain_ec.h5 --autosave-dir=proteinbert_models/cluster-30 >& pretrain-30.log