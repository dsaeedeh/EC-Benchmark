import numpy as np
import pandas as pd
import h5py
import shared_utils.util as sh
import csv
import os
import argparse

def create_h5_dataset(data_path, output_h5_file_path, records_limit = None, save_chunk_size=10000, verbose=True, log_progress_every=10000):
    data_path_c = data_path + '/pretrain_ec.feather'
    data = pd.read_feather(data_path_c)

    n_seqs = data.shape[0]
    
    if verbose:
        sh.log('Will create an h5 dataset of %d sequences.' % n_seqs)

    output_h5_file_path = output_h5_file_path + '/pretrain_ec.h5'
    with h5py.File(output_h5_file_path, 'w') as h5f:
        uniprot_ids = h5f.create_dataset('uniprot_ids', shape=(n_seqs,), dtype=h5py.string_dtype())
        seqs = h5f.create_dataset('seqs', shape=(n_seqs,), dtype=h5py.string_dtype())
        seq_lengths = h5f.create_dataset('seq_lengths', shape=(n_seqs,), dtype=np.int32)

        start_index = 0

        for seqs_and_annotations_chunk in sh.to_chunks(load_seqs(data, records_limit = records_limit, verbose = verbose, log_progress_every = log_progress_every), save_chunk_size):
            end_index = start_index + len(seqs_and_annotations_chunk)
            uniprot_id_chunk, seq_chunk = map(list, zip(*seqs_and_annotations_chunk))

            uniprot_ids[start_index:end_index] = uniprot_id_chunk
            seqs[start_index:end_index] = seq_chunk
            seq_lengths[start_index:end_index] = list(map(len, seq_chunk))
            start_index = end_index

    if verbose:
        sh.log('Done.')


def load_seqs(data, records_limit = None, verbose=True, log_progress_every = 10000):

    if verbose:
        sh.log('Loading %s records...' % ('all' if records_limit is None else records_limit))

    for i, row in enumerate(data.iterrows()):
        if verbose and i % log_progress_every == 0:
            sh.log('%d/%d' % (i, len(data)), end = '\r')

        yield row[1]['id'], row[1]['seq']

    if verbose:
        sh.log('Finished. %d records.' % (len(data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.h5 pretrain data creation')
    parser.add_argument('--data_path', type=str, default='/home/ceas/davoudis/paper/data', help='Path to the data')
    parser.add_argument('--output_h5_file_path', type=str, default='/home/ceas/davoudis/paper/data/cluster-0', help='Path to the output .h5 file')
    args = parser.parse_args()
    
    create_h5_dataset(args.data_path, args.output_h5_file_path)
