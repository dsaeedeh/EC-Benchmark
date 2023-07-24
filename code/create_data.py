import pandas as pd
import json
import fasta2csv.converter
from Bio import SeqIO
import argparse


def convert_fasta_to_csv(fasta_path, csv_path):
    fasta2csv.converter.convert(fasta_path, csv_path)

def create_clusters(cluster_path_100, cluster_path_90, cluster_path_70, cluster_path_50, cluster_path_30):
    # read cluster files
    cluster_100 = pd.read_csv(cluster_path_100, sep='\t', header=None)  
    cluster_100.columns = ['representative', 'member']
    cluster_100 = cluster_100.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_100.to_csv('data/cluster-100/clusterRes_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_90 = pd.read_csv(cluster_path_90, sep='\t', header=None)
    cluster_90.columns = ['representative', 'member']
    cluster_90 = cluster_90.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_90.to_csv('data/cluster-90/clusterRes_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_70 = pd.read_csv(cluster_path_70, sep='\t', header=None)
    cluster_70.columns = ['representative', 'member']
    cluster_70 = cluster_70.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_70.to_csv('data/cluster-70/clusterRes_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_50 = pd.read_csv(cluster_path_50, sep='\t', header=None)
    cluster_50.columns = ['representative', 'member']
    cluster_50 = cluster_50.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_50.to_csv('data/cluster-50/clusterRes_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_30 = pd.read_csv(cluster_path_30, sep='\t', header=None)
    cluster_30.columns = ['representative', 'member']
    cluster_30 = cluster_30.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_30.to_csv('data/cluster-30/clusterRes_cluster_final.tsv', sep='\t', index=False, header=False)

             
# Create fine-tuning data
'''
1. remove similar sequences from train data based on the clustering result
2. Add 3d information for train and test data
'''
# Create pretrain data
'''
remove EC numbers from pretrain data for the sequences that are similar to the sequences in test data based on the clustering result
'''

def create_data(pretrain_ec_path, train_ec_path, test_ec_path, train_3d_path, test_3d_path, info_file_path):

    train = pd.read_feather(train_ec_path)
    test = pd.read_feather(test_ec_path)
    test_ids = list(test['id'])
    pretrain = pd.read_feather(pretrain_ec_path)
    train_3d_id = [record.id for record in SeqIO.parse(train_3d_path, 'fasta')]
    test_3d_id = [record.id for record in SeqIO.parse(test_3d_path, 'fasta')]
    with open(info_file_path, 'r') as f:
            info = json.load(f)

    cluster_paths = ['data/cluster-100', 'data/cluster-90', 'data/cluster-70', 'data/cluster-50', 'data/cluster-30']
    t_list = [1.0, 0.9, 0.7, 0.5, 0.3]
    ids_to_remove = []

    for threshold in t_list:
        if threshold == 1.0:
            path = cluster_paths[0]
            ids_to_remove_100 = []
            clustering_path = path + '/clusterRes_cluster_final.tsv'
            clusters_100 = pd.read_csv(clustering_path, sep='\t', header=None)
            for i in range(clusters_100.shape[0]):
                cluster_list = []
                cluster_list.append(clusters_100.iloc[i, 0])
                cluster_list.extend(clusters_100.iloc[i, 1].split(','))
                if len(set(cluster_list).intersection(set(test_ids))) > 0:
                    ids_to_remove_100.extend(cluster_list)
            ids_to_remove.extend(ids_to_remove_100)
            del clusters_100

        elif threshold == 0.9:
            path = cluster_paths[1]
            ids_to_remove_90 = []
            clustering_path = path + '/clusterRes_cluster_final.tsv'
            clusters_90 = pd.read_csv(clustering_path, sep='\t', header=None)
            for i in range(clusters_90.shape[0]):
                cluster_list = []
                cluster_list.append(clusters_90.iloc[i, 0])
                cluster_list.extend(clusters_90.iloc[i, 1].split(','))
                if len(set(cluster_list).intersection(set(test_ids))) > 0:
                    ids_to_remove_90.extend(cluster_list)
            ids_to_remove.extend(ids_to_remove_90)
            del clusters_90

        elif threshold == 0.7:
            path = cluster_paths[2]
            ids_to_remove_70 = []
            clustering_path = path + '/clusterRes_cluster_final.tsv'
            clusters_70 = pd.read_csv(clustering_path, sep='\t', header=None)
            for i in range(clusters_70.shape[0]):
                cluster_list = []
                cluster_list.append(clusters_70.iloc[i, 0])
                cluster_list.extend(clusters_70.iloc[i, 1].split(','))
                if len(set(cluster_list).intersection(set(test_ids))) > 0:
                    ids_to_remove_70.extend(cluster_list)
            ids_to_remove.extend(ids_to_remove_70)
            del clusters_70

        elif threshold == 0.5:
            path = cluster_paths[3]
            ids_to_remove_50 = []
            clustering_path = path + '/clusterRes_cluster_final.tsv'
            clusters_50 = pd.read_csv(clustering_path, sep='\t', header=None)
            for i in range(clusters_50.shape[0]):
                cluster_list = []
                cluster_list.append(clusters_50.iloc[i, 0])
                cluster_list.extend(clusters_50.iloc[i, 1].split(','))
                if len(set(cluster_list).intersection(set(test_ids))) > 0:
                    ids_to_remove_50.extend(cluster_list)
            ids_to_remove.extend(ids_to_remove_50)
            del clusters_50

        else:
            path = cluster_paths[4]
            ids_to_remove_30 = []
            clustering_path = path + '/clusterRes_cluster_final.tsv'
            clusters_30 = pd.read_csv(clustering_path, sep='\t', header=None)
            for i in range(clusters_30.shape[0]):
                cluster_list = []
                cluster_list.append(clusters_30.iloc[i, 0])
                cluster_list.extend(clusters_30.iloc[i, 1].split(','))
                if len(set(cluster_list).intersection(set(test_ids))) > 0:
                    ids_to_remove_30.extend(cluster_list)
            ids_to_remove.extend(ids_to_remove_30)
            del clusters_30
            
        # Step 1
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
        
        for i in range(train.shape[0]):
            if train.iloc[i, 0] in info.keys():
                train_info_list.append(info[train.iloc[i, 0]])
        for i in range(test.shape[0]):
            if test.iloc[i, 0] in info.keys():
                test_info_list.append(info[test.iloc[i, 0]])
    
        train['3d_info'] = train_info_list
        test['3d_info'] = test_info_list
        train_path = path + '/train_ec_3d.csv'
        train.to_csv(train_path, index=False)
        print('train size: ', train.shape)
        test_path = path + '/test_ec_3d.csv'
        test.to_csv(test_path, index=False)
        print('test size: ', test.shape)
        del train, test, train_3d_id, test_3d_id, train_info_list, test_info_list

        for i in range(pretrain.shape[0]):
            if pretrain.iloc[i, 0] in ids_to_remove:
                pretrain['ec_number'][i] = '-'

        pretrain_path = path + '/pretrain_ec.csv'
        pretrain.to_csv(pretrain_path, index=False)
        print('pretrain size: ', pretrain.shape)
        del pretrain


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning data creation')
    parser.add_argument('--pretrain_ec_path', type=str, default='data/pretrain_ec.feather', help='Path to pretrain ec data')
    parser.add_argument('--train_ec_path', type=str, default='data/train_ec.feather', help='Path to train ec data')
    parser.add_argument('--test_ec_path', type=str, default='data/test_ec.feather', help='Path to test ec data')
    parser.add_argument('--train_3d_path', type=str, default='data/train_having_3d.fasta', help='Path to train 3d data')
    parser.add_argument('--test_3d_path', type=str, default='data/test_having_3d.fasta', help='Path to test 3d data')
    parser.add_argument('--info_file_path', type=str, default='data/swissprot_coordinates.json', help='Path to all 3d coordinates file')
    args = parser.parse_args()

    #create_clusters(cluster_path_100='data/cluster-100/clusterRes_cluster.tsv', cluster_path_90='data/cluster-90/clusterRes_cluster.tsv', cluster_path_70='data/cluster-70/clusterRes_cluster.tsv', cluster_path_50='data/cluster-50/clusterRes_cluster.tsv', cluster_path_30='data/cluster-30/clusterRes_cluster.tsv')
    create_data(pretrain_ec_path=args.pretrain_ec_path, train_ec_path=args.train_ec_path, test_ec_path=args.test_ec_path, train_3d_path=args.train_3d_path, test_3d_path=args.test_3d_path, info_file_path=args.info_file_path)    
