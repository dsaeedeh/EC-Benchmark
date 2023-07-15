#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

python code/download_data.py 2023 01 2018 02