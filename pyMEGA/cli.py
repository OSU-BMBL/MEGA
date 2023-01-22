"""
Main CLI entry file.
"""

import sys
import argparse
import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
# from kneed import KneeLocator   # 这个在后面并没有用到
import torch.utils.data as data

from warnings import filterwarnings

filterwarnings("ignore")
from torch_geometric.data import Data
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy import interp


# pyHGT
from .pyHGT.utils import *
from .pyHGT.data import *
from .pyHGT.model import *


def main():
    """The main routine."""
    print("This is the main routine.")
    seed = 0

    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="Training GNN on species_sample graph")
    parser.add_argument("-epoch", type=int, default=500)
    # Result
    parser.add_argument(
        "-result_dir",
        type=str,
        default=r"/fs/ess/PCON0022/yuhan/HGT/IOM_3/co_result/",
        help="The address for storing the models and optimization results.",
    )
    parser.add_argument(
        "-input_dir1", default=None, help="The address of abundance matrix."
    )
    parser.add_argument(
        "-input_dir2", default=None, help="The address of metabolic matrix."
    )
    parser.add_argument(
        "-input_dir3", default=None, help="The address of phylogenetic matrix."
    )
    # Feature extration
    parser.add_argument(
        "-num", type=float, default=0.9, help="the num of training data"
    )
    parser.add_argument(
        "-reduction",
        type=str,
        default="AE",
        help="the method for feature extraction, pca, raw",
    )

    parser.add_argument(
        "-in_dim", type=int, default=256, help="Number of hidden dimension (AE)"
    )

    # GAE
    parser.add_argument(
        "-kl_coef",
        type=float,
        default=0.00005,  # KL co-efficient
        help="coefficient of regular term",
    )
    parser.add_argument(
        "-gamma", type=float, default=2.5, help="coefficient of focal loss"
    )
    parser.add_argument("-lr", type=float, default=0.003, help="learning rate")
    parser.add_argument(
        "-n_hid", type=int, default=128, help="Number of hidden dimension"
    )
    parser.add_argument(
        "-n_heads", type=int, default=8, help="Number of attention head"
    )
    parser.add_argument("-n_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("-dropout", type=float, default=0, help="Dropout ratio")
    parser.add_argument(
        "-layer_type", type=str, default="hgt", help="the layer type for GAE"
    )
    parser.add_argument("-loss", type=str, default="cross", help="the loss for GAE")

    parser.add_argument("-cuda", type=int, default=1, help="cuda 0 use GPU0 else cpu ")

    parser.add_argument("-rep", type=str, default="iT", help="precision truncation")

    parser.add_argument(
        "-AEtype",
        type=int,
        default=1,
        help="AEtype1: embedding node autoencoder 2:HGT node autoencode",
    )

    # the weight of each cancer type in loss function

    # parser.add_argument('--weight', type = float, nargs = '+', default=[0.755, 0.912, 0.696, 0.882, 0.755],
    #                   help='weight of each class in loss function')

    args = parser.parse_args()
    # fish = tank_to_fish.get(args.tank, "")
    print(args)
    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do. Return values are exit codes.


if __name__ == "__main__":
    main()
