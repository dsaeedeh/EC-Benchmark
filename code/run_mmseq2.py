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
all: 109417142
----------------------------------------
train with 3d structure: 534096
test with 3d structure: 1536
'''
def count_protein_number(fasta_file):
    count = 0
    for record in SeqIO.parse(fasta_file, 'fasta'):
        count += 1
    return count

count_pretrain = count_protein_number('data/pretrain.fasta')
count_train = count_protein_number('data/train.fasta')
count_test = count_protein_number('data/test.fasta')
count_all = count_protein_number('data/all.fasta')

print(f'pretrain: {count_pretrain}')
print(f'train: {count_train}')
print(f'test: {count_test}')
print(f'sum: {count_pretrain+count_train+count_test}')
print(f'all: {count_all}')

n_train = count_protein_number('data/train_having_3d.fasta')
n_test = count_protein_number('data/test_having_3d.fasta')
print(f'train with 3d structure: {n_train}')
print(f'test with 3d structure: {n_test}')