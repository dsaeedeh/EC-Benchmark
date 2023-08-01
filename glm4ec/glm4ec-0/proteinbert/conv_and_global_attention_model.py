import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

    
def create_model(seq_len, vocab_size, d_hidden_seq = 128, n_blocks = 6, conv_kernel_size = 9, wide_conv_dilation_rate = 5, activation = 'gelu'):
    
    '''
    seq_len is required to create the model, but all the weights are independent of the length and can be re-used with
    different lengths.
    '''
    
    input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
    
    hidden_seq = keras.layers.Embedding(vocab_size, d_hidden_seq, name = 'embedding-seq-input')(input_seq)
    
    for block_index in range(1, n_blocks + 1):
        
        narrow_conv_seq = keras.layers.Conv1D(filters = d_hidden_seq, kernel_size = conv_kernel_size, strides = 1, \
                padding = 'same', dilation_rate = 1, activation = activation, name = 'narrow-conv-block%d' % block_index)(hidden_seq)
        wide_conv_seq = keras.layers.Conv1D(filters = d_hidden_seq, kernel_size = conv_kernel_size, strides = 1, \
                padding = 'same', dilation_rate = wide_conv_dilation_rate, activation = activation, name = 'wide-conv-block%d' % \
                block_index)(hidden_seq)
        
        hidden_seq = keras.layers.Add(name = 'seq-merge1-block%d' % block_index)([hidden_seq, narrow_conv_seq, wide_conv_seq])
        hidden_seq = keras.layers.LayerNormalization(name = 'seq-merge1-norm-block%d' % block_index)(hidden_seq)
        
        dense_seq = keras.layers.Dense(d_hidden_seq, activation = activation, name = 'seq-dense-block%d' % block_index)(hidden_seq)
        hidden_seq = keras.layers.Add(name = 'seq-merge2-block%d' % block_index)([hidden_seq, dense_seq])
        hidden_seq = keras.layers.LayerNormalization(name = 'seq-merge2-norm-block%d' % block_index)(hidden_seq)
        
    output_seq = keras.layers.Dense(vocab_size, activation = 'softmax', name = 'output-seq')(hidden_seq)

    return keras.models.Model(inputs = input_seq, outputs = output_seq)
    
def get_model_with_hidden_layers_as_outputs(model):
    
    _, seq_len, _ = model.outputs[0].shape
    
    seq_layers = [layer.output for layer in model.layers if len(layer.output.shape) == 3 and \
            tuple(layer.output.shape)[:2] == (None, seq_len) and (layer.name in ['input-seq-encoding', 'dense-seq-input', 'output-seq'] or \
            isinstance(layer, keras.layers.LayerNormalization))]
  
    concatenated_seq_output = keras.layers.Concatenate(name = 'all-seq-layers')(seq_layers)
    
    return keras.models.Model(inputs = model.inputs, outputs = concatenated_seq_output)
    