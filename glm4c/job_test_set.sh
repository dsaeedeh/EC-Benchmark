#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00                  # Max wall-clock time 1 day 1 hour
#SBATCH --ntasks=1                         # number of cores

# run tensorflow in singularity container
# redirect output to a file so that it can be inspected before the end of the job
singularity exec /home/ceas/davoudis/tensorflow_nvidia.sif python3 ./bin/set_h5_testset.py --h5-dataset-file=pretrain_twogeomes_removed_similars.h5 >& h5-dataset-file-test.log