from Bio import SeqIO
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

# Count the number of proteins in a fasta file using biopython
'''
pretrain: 108857716
train: 556825
test: 2601
sum: 109417142
all: 109417142
'''
def count_protein_number(fasta_file):
    count = 0
    for record in SeqIO.parse(fasta_file, 'fasta'):
        count += 1
    return count

