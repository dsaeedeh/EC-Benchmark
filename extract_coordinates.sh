#!/bin/bash
#SBATCH --job-name=extract_coordinates
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

# run tensorflow in singularity container
# redirect output to a file so that it can be inspected before the end of the job
python3 extract_coordinates.py data/swissprot_pdb_v4 >& extract_coordinates.log