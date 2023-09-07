import os
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

from tensorflow import keras

from . import conv_and_global_attention_model
from .model_generation import load_pretrained_model_from_dump

DEFAULT_LOCAL_MODEL_DUMP_DIR = 'proteinbert_models/cluster-30'
DEFAULT_LOCAL_MODEL_DUMP_FILE_NAME = 'epoch_55000_sample_59630000.pkl'

def load_pretrained_model(local_model_dump_dir=DEFAULT_LOCAL_MODEL_DUMP_DIR,
                          local_model_dump_file_name=DEFAULT_LOCAL_MODEL_DUMP_FILE_NAME,
                          download_model_dump_if_not_exists=True,
                          validate_downloading=True,
                          create_model_function=conv_and_global_attention_model.create_model, create_model_kwargs={},
                          optimizer_class=keras.optimizers.Adam, lr=2e-04,
                          other_optimizer_kwargs={}, annots_loss_weight=1, load_optimizer_weights=False):
    local_model_dump_dir = os.path.expanduser(local_model_dump_dir)
    dump_file_path = os.path.join(local_model_dump_dir, local_model_dump_file_name)


    return load_pretrained_model_from_dump(dump_file_path, create_model_function,
                                           create_model_kwargs=create_model_kwargs, optimizer_class=optimizer_class,
                                           lr=lr,
                                           other_optimizer_kwargs=other_optimizer_kwargs,
                                           annots_loss_weight=annots_loss_weight,
                                           load_optimizer_weights=load_optimizer_weights)
