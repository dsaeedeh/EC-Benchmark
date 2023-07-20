#!/bin/bash
#SBATCH --job-name=model-featurized
#SBATCH --partition=math-alderaan-gpu
#SBATCH --nodelist=math-alderaan-h02
#SBATCH --time=7-00:00:00                  # Max wall-clock time 1 day 1 hour
#SBATCH --ntasks=1                         # number of cores
#SBATCH --gres=gpu:a100:1  

# run tensorflow in singularity container
# redirect output to a file so that it can be inspected before the end of the job
export HDF5_USE_FILE_LOCKING=FALSE
singularity exec /home/davoudis/tensorflow_nvidia.sif python3 bin/pretrain_proteinbert.py --dataset-file=pretrain_unfeaturized.h5 --resume-from=proteinbert_models/epoch_240180_sample_63600000.pkl --autosave-dir=proteinbert_models >& pretrain.log