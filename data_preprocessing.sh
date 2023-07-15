#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

python3 code/data_preprocessing.py --pretrain_path data/pretrain.fasta --train_path data/train.fasta --test_path data/test.fasta --price_path data/price.fasta --info_file_path data/swissprot_coordinates.json &> data_coordinates.log