import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
sys.path.append(os.getcwd())

from functools import reduce
from ECRECer.tools import filetool as ftool
from ECRECer.tools import exact_ec_from_uniprot as exactec
from ECRECer.tools import funclib
from ECRECer.tools import minitools as mtool
from ECRECer.tools import embdding_onehot as onehotebd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# Create .tsv file of all files in data folder with .data.gz extension
for file in os.path.listdir('data'):
    if file.endswith('.data.gz'):
        exactec.run_exact_task(infile=f'data/{file}', outfile=f'data/{file}.tsv')