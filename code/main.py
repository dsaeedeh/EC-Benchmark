# this is the main file to run models
'''
List of models:
1. GLM4EC
2. DeepEC
3. ECPred
4. UDSMProt
5. Penlight
6. ProteInfer
7. ECRECer
8. CLEAN
9. BLASTp
10. CatFam
11. PRIAMv2
'''

# import models
from models import GLM4EC
from models import DeepEC
from models import ECPred
from models import UDSMProt
from models import Penlight
from models import ProteInfer
from models import ECRECer
from models import CLEAN
from models import BLASTp
from models import CatFam
from models import PRIAMv2

# import utils
from utils import get_data_for_pretrain
from utils import get_data_for_test
from utils import get_data_for_train

# import metrics
from metrics import get_metrics


def main():

