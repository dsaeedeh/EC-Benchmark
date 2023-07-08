#!/bin/bash
#SBATCH --job-name=extract_coordinates
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

python3 code/download_data.py 2023 01 2018 02 >& download_data.log