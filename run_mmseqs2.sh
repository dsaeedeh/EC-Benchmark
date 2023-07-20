#!/bin/bash
#SBATCH --job-name=run_mmseqs2
#SBATCH --partition=math-alderaan
#SBATCH --time=7-00:00:00  

# Use mmseqs2 to cluster all sequences into families with 30% - 50% - 70% - 90% identity; or any other identity threshold you want to try.
conda install -c conda-forge -c bioconda mmseqs2
cd data
awk 1 *.fasta > all.fasta
mmseqs easy-cluster all_having_3d.fasta clusterRes clusters --min-seq-id 0.9 -c 0.8 --cov-mode 1
mmseqs easy-cluster cluster-90/clusterRes_rep_seq.fasta clusterRes clusters --min-seq-id 0.7 -c 0.8 --cov-mode 1
mmseqs easy-cluster cluster-70/clusterRes_rep_seq.fasta clusterRes clusters --min-seq-id 0.5 -c 0.8 --cov-mode 1
mmseqs easy-cluster cluster-50/clusterRes_rep_seq.fasta clusterRes clusters --min-seq-id 0.3 -c 0.8 --cov-mode 1