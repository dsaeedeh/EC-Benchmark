import pandas as pd
import json
import fasta2csv.converter
from Bio import SeqIO
import argparse

# Create fine-tuning data
'''
1. remove similar sequences from train data based on the clustering result
2. Add 3d information for train and test data
'''
def convert_fasta_to_csv(fasta_path, csv_path):
    fasta2csv.converter.convert(fasta_path, csv_path)


def create_data(pretrain_ec_path, train_ec_path, test_ec_path, train_3d_path, test_3d_path, info_file_path, clustering_path):

    train = pd.read_feather(train_ec_path)
    test = pd.read_feather(test_ec_path)
    train_3d_id = [record.id for record in SeqIO.parse(train_3d_path, 'fasta')]
    test_3d_id = [record.id for record in SeqIO.parse(test_3d_path, 'fasta')]

    clusters = pd.read_csv(clustering_path, sep='\t', header=None)
    # add column names to the clusters dataframe
    clusters.columns = ['representative', 'member']
    # concat member values together for each representative
    clusters = clusters.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    print('clusters-30 size: ', clusters.shape)
    
    # Step 1
    ids_to_remove = []
    test_ids = list(test['id'])

    for i in range(clusters.shape[0]):
        cluster_list = []
        cluster_list.append(clusters.iloc[i, 0])
        cluster_list.extend(clusters.iloc[i, 1].split(','))
        if len(set(cluster_list).intersection(set(test_ids))) > 0:
            ids_to_remove.extend(cluster_list)

    ids_to_remove = list(set(ids_to_remove))
    train = train[~train['id'].isin(ids_to_remove)]
    train.reset_index(drop=True, inplace=True)
    train = train[train['id'].isin(train_3d_id)]
    train.reset_index(drop=True, inplace=True)
    test = test[test['id'].isin(test_3d_id)]
    test.reset_index(drop=True, inplace=True)
    
    # Step 2
    train_info_list = []
    test_info_list = []
    with open(info_file_path, 'r') as f:
        info = json.load(f)
        for i in range(train.shape[0]):
            if train.iloc[i, 0] in info.keys():
                train_info_list.append(info[train.iloc[i, 0]])
        for i in range(test.shape[0]):
            if test.iloc[i, 0] in info.keys():
                test_info_list.append(info[test.iloc[i, 0]])
    
    train['3d_info'] = train_info_list
    test['3d_info'] = test_info_list
    train.to_csv('data/cluster-30/train_ec_3d.csv', index=False)
    test.to_csv('data/cluster-30/test_ec_3d.csv', index=False)

    pretrain = pd.read_feather(pretrain_ec_path)
    for i in range(pretrain.shape[0]):
        if pretrain.iloc[i, 0] in ids_to_remove:
            pretrain['ec_number'][i] = '-'

    pretrain.to_csv('data/cluster-30/pretrain_ec.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning data creation')
    parser.add_argument('--pretrain_ec_path', type=str, default='data/pretrain_ec.feather', help='Path to pretrain ec data')
    parser.add_argument('--train_ec_path', type=str, default='data/train_ec.feather', help='Path to train ec data')
    parser.add_argument('--test_ec_path', type=str, default='data/test_ec.feather', help='Path to test ec data')
    parser.add_argument('--train_3d_path', type=str, default='data/train_having_3d.fasta', help='Path to train 3d data')
    parser.add_argument('--test_3d_path', type=str, default='data/test_having_3d.fasta', help='Path to test 3d data')
    parser.add_argument('--info_file_path', type=str, default='data/swissprot_coordinates.json', help='Path to all 3d coordinates file')
    parser.add_argument('--clustering_path', type=str, default='data/cluster-30/clusterRes_cluster.tsv', help='Path to clustering file')
    args = parser.parse_args()

    create_data(pretrain_ec_path=args.pretrain_ec_path, train_ec_path=args.train_ec_path, test_ec_path=args.test_ec_path, train_3d_path=args.train_3d_path, test_3d_path=args.test_3d_path, info_file_path=args.info_file_path, clustering_path=args.clustering_path)    
