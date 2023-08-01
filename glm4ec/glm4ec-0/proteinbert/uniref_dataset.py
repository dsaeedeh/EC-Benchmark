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
    annotation = []

    for j in data['ec_number'].str.split(','):
        for i in j:
            annotation.append(i)

    annotation = list(set(annotation))
    annotation.remove('-')

    n_annotations = len(annotation)
    print('number of annotations:', n_annotations)

    dict_annotation = dict(zip(annotation, range(n_annotations)))

    with open(os.path.join(data_path, 'dict_annotations_pretrain.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_annotation.items():
            writer.writerow([key, value])

    if verbose:
        sh.log('Will create an h5 dataset of %d sequences.' % n_seqs)

    output_h5_file_path = output_h5_file_path + '/pretrain_ec.h5'
    with h5py.File(output_h5_file_path, 'w') as h5f:
        h5f.create_dataset('included_annotations', data=[a.encode('ascii') for a in annotation],
                           dtype=h5py.string_dtype())
        uniprot_ids = h5f.create_dataset('uniprot_ids', shape=(n_seqs,), dtype=h5py.string_dtype())
        seqs = h5f.create_dataset('seqs', shape=(n_seqs,), dtype=h5py.string_dtype())
        seq_lengths = h5f.create_dataset('seq_lengths', shape=(n_seqs,), dtype=np.int32)
        annotation_masks = h5f.create_dataset('annotation_masks', shape=(n_seqs, n_annotations), dtype=bool)
        dict_annotation = dict(zip(annotation, range(n_annotations)))

        start_index = 0

        for seqs_and_annotations_chunk in sh.to_chunks(load_seqs_and_annotations(data, records_limit = records_limit, verbose = verbose, log_progress_every = log_progress_every), save_chunk_size):
            end_index = start_index + len(seqs_and_annotations_chunk)
            uniprot_id_chunk, seq_chunk, annotation_indices_chunk = map(list, zip(*seqs_and_annotations_chunk))

            uniprot_ids[start_index:end_index] = uniprot_id_chunk
            seqs[start_index:end_index] = seq_chunk
            seq_lengths[start_index:end_index] = list(map(len, seq_chunk))
            annotation_masks[start_index:end_index, :] = _encode_annotations_as_a_binary_matrix(
                annotation_indices_chunk, dict_annotation)
            start_index = end_index

    if verbose:
        sh.log('Done.')


def load_seqs_and_annotations(data, records_limit = None, verbose=True, log_progress_every = 10000):

    if verbose:
        sh.log('Loading %s records...' % ('all' if records_limit is None else records_limit))

    for i, row in enumerate(data.iterrows()):
        if verbose and i % log_progress_every == 0:
            sh.log('%d/%d' % (i, len(data)), end = '\r')

        yield row[1]['id'], row[1]['seq'], row[1]['ec_number']

    if verbose:
        sh.log('Finished. %d records.' % (len(data)))


def _encode_annotations_as_a_binary_matrix(records_annotations, annotation_to_index):
    annotation_masks = np.zeros((len(records_annotations), len(annotation_to_index)), dtype=bool)

    for i, record_annotations in enumerate(records_annotations):
        if record_annotations != '-':
            for j in record_annotations.split(','):
                v = annotation_to_index.get(j)
                annotation_masks[i, v] = True
        # print the number of True values in each row
        # print(np.count_nonzero(annotation_masks[i, :]))

    return annotation_masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.h5 pretrain data creation')
    parser.add_argument('--data_path', type=str, default='/home/ceas/davoudis/paper/data/cluster-100', help='Path to the data')
    parser.add_argument('--output_h5_file_path', type=str, default='/home/ceas/davoudis/paper/data/cluster-100', help='Path to the output .h5 file')
    args = parser.parse_args()
    
    create_h5_dataset(args.data_path, args.output_h5_file_path)
