import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
sys.path.append(os.getcwd())
import argparse
from functools import reduce
from ECRECer.tools import filetool as ftool
from ECRECer.tools import exact_ec_from_uniprot as exactec
from ECRECer.tools import funclib
from ECRECer.tools import minitools as mtool
from ECRECer.tools import embdding_onehot as onehotebd
from pandarallel import pandarallel
from Bio import SeqIO
import json

pandarallel.initialize(progress_bar=True)


def create_tsv_from_data():
    for file in os.listdir('data'):
        if file.endswith('.data.gz'):
            exactec.run_exact_task(infile=f'data/{file}', outfile=f'data/{file}.tsv')

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def preprocessing(pretrain_path, train_path, test_path, price_path):
    # Read the price file
    price = pd.read_csv(price_path, header=0)
    price.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    price.reset_index(drop=True, inplace=True)
    price['ec_number'] = price.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    price['ec_number'] = price.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    price['functionCounts'] = price.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('price finished')
    price.to_feather('data/snp_price.feather')

    pretrain_data = pd.read_csv(pretrain_path, sep='\t', header=0)
    pretrain_data = mtool.convert_DF_dateTime(inputdf=pretrain_data)
    pretrain_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    pretrain_data.reset_index(drop=True, inplace=True)

    pretrain_data['ec_number'] = pretrain_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    pretrain_data['ec_number'] = pretrain_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    pretrain_data['functionCounts'] = pretrain_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('pretrain_data finished')
    pretrain_data.to_feather('data/snp_pretrain_data.feather')
    pretrain = pretrain_data.iloc[:,np.r_[0,2:8,10:12]]

    del pretrain_data

    train_data = pd.read_csv(train_path, sep='\t', header=0)
    train_data = mtool.convert_DF_dateTime(inputdf=train_data)
    train_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    train_data['ec_number'] = train_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    train_data['ec_number'] = train_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    train_data['functionCounts'] = train_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('train_data finished')
    train_data.to_feather('data/snp_train_data.feather')
    train = train_data.iloc[:,np.r_[0,2:8,10:12]]

    del train_data

    test_data = pd.read_csv(test_path, sep='\t', header=0)
    test_data = mtool.convert_DF_dateTime(inputdf=test_data)
    test_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    test_data['ec_number'] = test_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    test_data['ec_number'] = test_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    test_data['functionCounts'] = test_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('test_data finished')
    test_data.to_feather('data/snp_test_data.feather')
    test = test_data.iloc[:,np.r_[0,2:8,10:12]]

    del test_data

    test =test[~test.seq.isin(train.seq)]
    test =test[~test.seq.isin(pretrain.seq)]
    test.reset_index(drop=True, inplace=True) 

    # Remove changed seqence in test set
    test = test[~test.id.isin(test.merge(pretrain, on='id', how='inner').id.values)]
    test = test[~test.id.isin(test.merge(train, on='id', how='inner').id.values)]
    test.reset_index(drop=True, inplace=True)

    pretrain = pretrain[~pretrain.id.isin(pretrain.merge(price, on='id', how='inner').id.values)]
    train = train[~train.id.isin(train.merge(price, on='id', how='inner').id.values)]
    train.reset_index(drop=True, inplace=True)
    pretrain.reset_index(drop=True, inplace=True)

    # Trim sequences
    with pd.option_context('mode.chained_assignment', None):
        pretrain.ec_number = pretrain.ec_number.parallel_apply(lambda x : str(x).strip())
        pretrain.seq = pretrain.seq.parallel_apply(lambda x : str(x).strip())

        train.ec_number = train.ec_number.parallel_apply(lambda x : str(x).strip())
        train.seq = train.seq.parallel_apply(lambda x : str(x).strip())

        test.ec_number = test.ec_number.parallel_apply(lambda x : str(x).strip())
        test.seq = test.seq.parallel_apply(lambda x : str(x).strip())

        price.ec_number = price.ec_number.parallel_apply(lambda x : str(x).strip())
        price.seq = price.seq.parallel_apply(lambda x : str(x).strip())
        
    pretrain.to_feather('data/pretrain.feather')
    train.to_feather('data/train.feather')
    test.to_feather('data/test.feather')

    # Create EC number prediction data
    pretrain = pretrain.iloc[:,np.r_[0,7,4]]
    train = train.iloc[:,np.r_[0,7,4]]
    test = test.iloc[:,np.r_[0,7,4]]

    pretrain.to_feather('data/pretrain_ec.feather')
    train.to_feather('data/train_ec.feather')
    test.to_feather('data/test_ec.feather')
    price.to_feather('data/price.feather')

    funclib.table2fasta(table=pretrain[['id', 'seq']], file_out='data/pretrain.fasta')
    funclib.table2fasta(table=train[['id', 'seq']], file_out='data/train.fasta')
    funclib.table2fasta(table=test[['id', 'seq']], file_out='data/test.fasta')
    funclib.table2fasta(table=price[['id', 'seq']], file_out='data/price.fasta')
 

# Check if we have 3d information for all train and test data
def check_3d_information(train_path, test_path, price_path, info_file_path):
    # get the protein ids from fasta files using biopython
    train_ids = [record.id for record in SeqIO.parse(train_path, 'fasta')]
    test_ids = [record.id for record in SeqIO.parse(test_path, 'fasta')]
    price_ids = [record.id for record in SeqIO.parse(price_path, 'fasta')]

    # get the ids from json info_file
    with open(info_file_path, 'r') as f:
        info = json.load(f)
        info_ids = list(info.keys())
    
    # check if all ids are in the info file
    train_ids = list(set(train_ids).intersection(set(info_ids)))
    test_ids = list(set(test_ids).intersection(set(info_ids)))
    price_ids = list(set(price_ids).intersection(set(info_ids)))
                     
    print(f'Number of train ids in info file: {len(train_ids)}')
    print(f'Number of test ids in info file: {len(test_ids)}')
    print(f'Number of price ids in info file: {len(price_ids)}')

    # exclude ids not in info file from train and test fasta data and save them to new files 
    SeqIO.write((record for record in SeqIO.parse(train_path, 'fasta') if record.id in train_ids), 'data/train_having_3d.fasta', 'fasta')
    SeqIO.write((record for record in SeqIO.parse(test_path, 'fasta') if record.id in test_ids), 'data/test_having_3d.fasta', 'fasta')
    SeqIO.write((record for record in SeqIO.parse(price_path, 'fasta') if record.id in price_ids), 'data/price_having_3d.fasta', 'fasta')


'''
pretrain: 108857557
train: 556822
test: 2601
price: 184
all: 109417164
----------------------------------------
Number of train ids with 3d info: 534093
Number of test ids with 3d info: 1536
Number of price ids with 3d info: 3
'''
def count_protein_number(fasta_file):
    count = 0
    for record in SeqIO.parse(fasta_file, 'fasta'):
        count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data merging script')
    parser.add_argument('--pretrain_path', type=str, help='Path to the pretrain data')
    parser.add_argument('--train_path', type=str, help='Path to the train data')
    parser.add_argument('--test_path', type=str, help='Path to the test data')
    parser.add_argument('--price_path', type=str, help='Path to the price data')
    parser.add_argument('--info_file_path', type=str, help='Path to the 3d coordinates file')
    args = parser.parse_args()

    # run following functions in order; comment out the functions that have been run
    create_tsv_from_data()
    preprocessing(pretrain_path=args.pretrain_path, train_path=args.train_path, test_path=args.test_path, price_path=args.price_path)
    check_3d_information(train_path=args.train_path, test_path=args.test_path, price_path= args.price_path, info_file_path=args.info_file_path)

    # Count the number of proteins in a fasta file using biopython
    count_pretrain = count_protein_number('data/pretrain.fasta')
    count_train = count_protein_number('data/train.fasta')
    count_test = count_protein_number('data/test.fasta')
    count_price = count_protein_number('data/price.fasta')
    count_all = count_protein_number('data/all.fasta')

    print(f'pretrain: {count_pretrain}')
    print(f'train: {count_train}')
    print(f'test: {count_test}')
    print(f'price: {count_price}')
    print(f'all: {count_all}')

    


