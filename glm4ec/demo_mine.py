import pandas as pd
import numpy as np
from IPython.display import display
import pickle
from tensorflow import keras
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, chunked_finetune, evaluate_by_len, log
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import os
import ast
import warnings
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import gc

warnings.filterwarnings("ignore")

BENCHMARKS_DIR = 'protein_benchmarks'
BENCHMARK_NAME = 'cluster-30'

np.random.seed(42)

train_set_file_path = os.path.join(BENCHMARKS_DIR, BENCHMARK_NAME, 'train_ec.csv')
train_set = pd.read_csv(train_set_file_path)
test_set = pd.read_csv(os.path.join(BENCHMARKS_DIR, BENCHMARK_NAME, 'test_ec.csv'))
price_set = pd.read_csv(os.path.join(BENCHMARKS_DIR, 'price.csv'))

print(f'{len(train_set)} training set records, {len(test_set)} test set records, {len(price_set)} price set records')

frame = [train_set, test_set, price_set]
final_data = pd.concat(frame)

annotation = []
for i in range(final_data.shape[0]):
    for j in final_data.iloc[i]['ec_number'].split(','):
        annotation.append(j)

UNIQUE_LABELS = list(set(annotation))
if '-' in UNIQUE_LABELS:
    UNIQUE_LABELS.remove('-')
print('number of annotations: ', len(UNIQUE_LABELS))
dict_annotation = dict(zip(UNIQUE_LABELS, range(len(UNIQUE_LABELS))))

with open(os.path.join('fine_tuning_results/cluster-30', 'dict_annotation.csv'), 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in dict_annotation.items():
       writer.writerow([key, value])


OUTPUT_TYPE = OutputType(False, 'MLC')

OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

gc.collect()

sq = 2048
read_chunk_seq_flag = True
n_final_epochs = 1
final_seq_len = 1024

pretrained_model_generator, input_encoder = load_pretrained_model()

model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
        get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(monitor='loss', patience = 2, restore_best_weights = True),
]

# +
if read_chunk_seq_flag == False:

    array_ecs_train = np.zeros((len(train_set), len(UNIQUE_LABELS)), dtype=int)

    for i in range(train_set.shape[0]):
        if train_set.iloc[i]['ec_number'] != '-':   
            for j in train_set.iloc[i]['ec_number'].split(','):
                v = dict_annotation.get(j)
                array_ecs_train[i, v] = 1

    train_set['label'] = array_ecs_train.tolist()

    print('we do not have chunked data')

    finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'],train_set['label'],
        seq_len = sq, batch_size = 32, max_epochs_per_stage = 40, lr = 1e-04, begin_with_frozen_pretrained_layers = True,
        lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = n_final_epochs, final_seq_len = final_seq_len, final_lr = 1e-05)
    fine_tuned_model = model_generator.create_model(sq)

    with open(os.path.join('fine_tuning_results/cluster-30', 'fine_tuned_model.pkl'), 'wb') as f:
        pickle.dump((fine_tuned_model.get_weights(), fine_tuned_model.optimizer.get_weights()), f) 
    
    array_ecs_test = np.zeros((len(test_set), len(UNIQUE_LABELS)), dtype=int)
    for i in range(test_set.shape[0]):
        if test_set.iloc[i]['ec_number'] != '-':
            for j in test_set.iloc[i]['ec_number'].split(','):
                v = dict_annotation.get(j)
                array_ecs_test[i, v] = 1
    test_set['label'] = array_ecs_test.tolist()
    
    array_ecs_price = np.zeros((len(price_set), len(UNIQUE_LABELS)), dtype=int)
    for i in range(price_set.shape[0]):
        if price_set.iloc[i]['ec_number'] != '-':
            for j in price_set.iloc[i]['ec_number'].split(','):
                v = dict_annotation.get(j)
                array_ecs_price[i, v] = 1
    price_set['label'] = array_ecs_price.tolist()

    results, confusion_matrix, y_true, y_pred = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'],
            start_seq_len = 512, start_batch_size = 32)

    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_pred.npy'), y_pred)
    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_true.npy'), y_true)
    
    results_p, confusion_matrix_p, y_true_p, y_pred_p = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, price_set['seq'], price_set['label'],
            start_seq_len = 512, start_batch_size = 32)
    
    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_pred_price.npy'), y_pred_p)
    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_true_price.npy'), y_true_p)

else:
    print('we have chunked data')
    chunked_finetune(dict_annotation, model_generator, input_encoder, OUTPUT_SPEC, train_set_file_path,
        seq_len = sq, batch_size = 32, max_epochs_per_stage = 40, lr = 1e-04, begin_with_frozen_pretrained_layers = True,
        lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = n_final_epochs, final_seq_len = final_seq_len, final_lr = 1e-05)

    fine_tuned_model = model_generator.create_model(sq)
    with open(os.path.join('fine_tuning_results/cluster-30', 'fine_tuned_model.pkl'), 'wb') as f:
        pickle.dump((fine_tuned_model.get_weights(), fine_tuned_model.optimizer.get_weights()), f)
    
    array_ecs_test = np.zeros((len(test_set), len(UNIQUE_LABELS)), dtype=int)
    for i in range(test_set.shape[0]):
        if test_set.iloc[i]['ec_number'] != '-':
            for j in test_set.iloc[i]['ec_number'].split(','):
                v = dict_annotation.get(j)
                array_ecs_test[i, v] = 1
    test_set['label'] = array_ecs_test.tolist()
    
    array_ecs_price = np.zeros((len(price_set), len(UNIQUE_LABELS)), dtype=int)
    for i in range(price_set.shape[0]):
        if price_set.iloc[i]['ec_number'] != '-':
            for j in price_set.iloc[i]['ec_number'].split(','):
                v = dict_annotation.get(j)
                array_ecs_price[i, v] = 1
    price_set['label'] = array_ecs_price.tolist()

    results, confusion_matrix, y_true, y_pred = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'],
            start_seq_len = 512, start_batch_size = 32)

    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_pred.npy'), y_pred)
    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_true.npy'), y_true)

    results_p, confusion_matrix_p, y_true_p, y_pred_p = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, price_set['seq'], price_set['label'],
            start_seq_len = 512, start_batch_size = 32)
    
    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_pred_price.npy'), y_pred_p)
    np.save(os.path.join('fine_tuning_results/cluster-30', 'y_true_price.npy'), y_true_p)
# -


