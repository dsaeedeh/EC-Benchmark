#!/bin/bash
#SBATCH --job-name=extract_coordinates
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

singularity exec /home/ceas/davoudis/tensorflow_nvidia.sif python3 code/create_finetuning_data.py &> data_create_finetuning.log