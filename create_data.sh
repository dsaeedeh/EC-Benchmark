#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

python3 code/create_data.py &> create_data.log