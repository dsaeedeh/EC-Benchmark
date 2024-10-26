#!/bin/bash
#SBATCH --job-name=extract_coordinates
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

python code/extract_coordinates.py data/swissprot_pdb_v4 >& extract_coordinates.log
# move the output files to the data directory
mv swissprot_coordinates.json data/