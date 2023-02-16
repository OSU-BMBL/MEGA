"""
Main CLI entry file.
"""
# pyHGT
from .pyHGT.utils import *
from .pyHGT.data import *
from .pyHGT.model import *

import sys
import math
import time
import argparse
import logging
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
import subprocess

# from kneed import KneeLocator   # 这个在后面并没有用到
import torch.utils.data as torchdata

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


try:
    from ._version import get_versions

    VERSION = get_versions()["version"]
except:
    VERSION = "?.?.?"

LOGGER = logging.getLogger(__name__)


def load_data(path1, path2, sep, col_name, row_name):
    """
    Load a abundance matrix from a CSV file and return various components of the matrix as separate variables.

    Parameters:
        path1 (str): The file path of the CSV file containing the abundance matrix.
        path2 (str): The file path of the CSV file containing sample cancer labels.
        sep (str, optional): The delimiter used in the CSV file. Default is ','.
        col_name (bool, optional): If True, the first column of the CSV file contains column names. Default is False.
        row_name (bool, optional): If True, the first row of the CSV file contains row names. Default is False.

    Returns:
        A tuple of the following variables:
        gene_cell_matrix1 (pandas.DataFrame): The abundance matrix after cleaning and filtering.
        gene_cell_matrix (pandas.DataFrame): The original abundance matrix.
        cell_label (pandas.DataFrame): The cell labels.
        gene_cell (numpy.ndarray): The abundance matrix as a NumPy array.
        gene_name (numpy.ndarray): The gene names.
        cell_name (numpy.ndarray): The cell names.

    """
    if col_name is True and row_name is True:
        gene_cell_matrix = pd.read_csv(path1, sep=sep, index_col=0)
        cell_label = pd.read_csv(path2, sep=sep, index_col=0)
    gene_cell_matrix1 = gene_cell_matrix.dropna(axis=0)
    gene_cell_matrix = gene_cell_matrix1
    gene_cell = gene_cell_matrix.values
    gene_name = gene_cell_matrix.index.values
    cell_name = gene_cell_matrix.columns.values

    return (
        gene_cell_matrix1,
        gene_cell_matrix,
        cell_label,
        gene_cell,
        gene_name,
        cell_name,
    )


def split_cell_train_test(args, data, name):
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(args.seed)
    # permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(np.arange(len(data)))
    # test_ratio为测试集所占的百分比
    train_set_size = int(len(data) * args.num)
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size : len(data)]
    train_data = []
    train_name = []
    test_data = []
    test_name = []
    for i in train_indices:
        train_data.append(data[i])
        train_name.append(name[i])
    for i in test_indices:
        test_data.append(data[i])
        test_name.append(name[i])
    return train_data, test_data, train_name, test_name


def shuffle(x1, y1, z1):
    x = []
    y = []
    z = []
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    for i in index:
        x.append(x1[i])
        y.append(y1[i])
        z.append(z1[i])
    return x, y, z


class AE(nn.Module):
    def __init__(self, dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.tanh(self.fc1(x))  # originally relu in both encode and decode
        # h2 = F.tanh(self.fc2(h1))
        return F.tanh(self.fc2(h1))
        # return h1

    def decode(self, z):
        h3 = F.tanh(self.fc5(z))
        # h4 = F.tanh(self.fc5(h3))
        return F.tanh(self.fc6(h3))
        # return torch.relu(self.fc4(z))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):  # gamma
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert (
                len(alpha) == num_classes
            )  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (
                1 - alpha
            )  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds_softmax, preds_logsoft, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        self.alpha = self.alpha.to(preds_softmax.device)
        # preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        # preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(
            1, labels.view(-1, 1)
        )  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        # preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        # ce_loss1 = torch.nn.NLLLoss()
        # ce_loss = ce_loss1(preds_logsoft, labels)
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(
            torch.pow((1 - preds_softmax), self.gamma), preds_logsoft
        )  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(
        TPR=tpr, FPR=fpr, threshold=thresholds
    )
    return fpr, tpr, roc_auc, optimal_th, optimal_point


def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def check_input_params(args):
    # input is csv

    # input and metadata label match

    # input row is species ID

    # output dir exist

    return True


def get_metabolic_matrix(input, out_dir, input_db="NJS16_metabolic_relation.txt"):
    df = pd.read_csv(input, sep=",", index_col=0)
    LOGGER.info(f"Number of species: {df.shape[0]}. Number of samples: {df.shape[1]}")

    species_list = df.index.values.tolist()
    species_list = map(
        lambda x: str(x), species_list
    )  # transfer the species id into str

    file_name = (input).split("/")[-1].split(".")[0]
    # 1, 2, 4 will remove
    file_1 = out_dir + "/" + file_name + "_njs16.csv"
    file_2 = out_dir + "/" + file_name + "_njs16_norm.txt"
    file_3 = out_dir + "/" + file_name + "_metabolic_matrix.csv"
    file_4 = out_dir + "/" + file_name + "_metabolic_relation.tsv"
    file_5 = out_dir + "/" + file_name + "_species_list.tsv"

    # Need to be checked
    df_njs16 = pd.read_csv(input_db, sep="\t")

    # obtain the contained species and metabolic compound in NJS16
    df_njs16 = df_njs16[df_njs16["taxonomy ID"].isin(species_list)]

    df_njs16.to_csv(file_1, sep="\t", header=0, index=0)

    # normalize the data: each compound is represented as a row
    f = open(file_1)
    w = open(file_2, "w")
    y = f.readline()
    while y:
        y = y.strip()
        lis = y.split("\t")
        # print (lis)
        if "&" in lis[0] and "," in lis[1]:
            lis_1 = lis[0].split("&")
            lis_2 = lis[1].split(",")
            for j in range(len(lis_2)):
                w.write(str(lis_1[0]) + "\t" + lis_2[j] + "\t" + lis[2] + "\n")
                w.write("Production (export)" + "\t" + lis_2[j] + "\t" + lis[2] + "\n")
        elif "&" in lis[0] and "," not in lis[1]:
            lis_1 = lis[0].split("&")
            w.write(str(lis_1[0]) + "\t" + lis[2] + "\t" + lis[2] + "\n")
            w.write("Production (export)" + "\t" + lis[2] + "\t" + lis[2] + "\n")

        elif "&" not in lis[0] and "," in lis[1]:
            lis_2 = lis[1].split(",")
            for j in range(len(lis_2)):
                w.write(str(lis[0] + "\t" + lis_2[j] + "\t" + lis[2] + "\n"))
        else:
            w.write(str(lis[0] + "\t" + lis[1] + "\t" + lis[2] + "\n"))
        y = f.readline()

    f.close()
    w.close()

    # all relations are considered as equivalent
    p = open(file_2)
    w = open(file_4, "w")

    dictionary = {}  # compound and species list. key is compound, and value is species
    x = p.readline()
    while x:
        x = x.strip()
        lis = x.split("\t")
        dictionary.setdefault(lis[1], []).append(lis[2])
        x = p.readline()

    list_species_pair = []  # move replicates of species-species relations
    for i, j in dictionary.items():
        if len(dictionary[i]) != 1:
            for k in range(len(j)):
                for k_1 in range(len(j)):
                    if set([j[k], j[k_1]]) not in list_species_pair:
                        list_species_pair.append(set([j[k], j[k_1]]))
                        w.write(str(j[k]) + "\t" + str(j[k_1]) + "\n")

    p.close()
    w.close()

    # transfer into a metabolic relation matrix
    species = df.index.values
    species_list = pd.DataFrame(species)
    species_list.to_csv(file_5, index=0, header=0)
    species = species.astype("int")
    species_metabolites = pd.read_csv(file_4, sep="\t", index_col=0)
    species1 = species_metabolites.index.values
    species2 = species_metabolites.values
    species2 = species2.flatten()
    matrix = np.zeros((len(species), len(species)))
    for i, j in enumerate(species):
        for m, n in enumerate(species1):
            if n == j:
                p = species2[m]
                q = np.argwhere(species == p)  # species.index(p)
                if i != q:
                    matrix[i, q] = 1
                    matrix[q, i] = 1

    df3 = pd.DataFrame(matrix, index=list(species), columns=list(species))
    df3.to_csv(file_3, sep=",")
    LOGGER.info(f"The metabolic relations among species are saved in {file_3}")

    os.remove(file_1)
    os.remove(file_2)
    os.remove(file_4)
    return file_5


def get_phylo_matrix(input1, input2, out_dir):
    # output file "*_metabolic_relation.tsv"
    file_name = (input2).split("/")[-1].split(".")[0]
    file_3 = out_dir + "/" + file_name + "_phy_matrix.csv"

    species_species = pd.read_csv(input2, sep=",", index_col=0)
    species = species_species.index.values
    species = species.astype("int")
    species_phylo = pd.read_csv(input1, sep="\t", index_col=0)
    genus = species_phylo.index.values
    genus_unrepeat = np.unique(genus)
    species3 = species_phylo.values
    matrix1 = np.zeros((len(species), len(species)))
    for i in genus_unrepeat:
        if i != 0:
            p = np.argwhere(genus == i).flatten()
            species4 = species3[p].flatten()
            q = []
            for n, m in enumerate(species):
                for k, h in enumerate(species4):
                    if h == m:
                        q.append(n)
            for j in q:
                for g in q:
                    if j != g:
                        matrix1[j, g] = 1
    df = pd.DataFrame(matrix1, index=list(species), columns=list(species))
    df.to_csv(file_3, sep=",")
    LOGGER.info(f"The phylogenic relations among species are saved in {file_3}")


def micah(args, out_dir, base_filename):
    # Set up folders
    file0 = (
        "pyMEGA_"
        + str(args.epoch)
        + "_kl_para_"
        + str(args.kl_coef)
        + "_gamma_"
        + str(args.gamma)
        + "_lr_"
        + str(args.lr)
    )

    att_file1 = out_dir + "/" + base_filename + "_attention.csv"
    path1 = out_dir + "/temp"
    metabolic_path = out_dir + "/" + base_filename + "_metabolic_matrix.csv"
    phylo_path = out_dir + "/" + base_filename + "_phy_matrix.csv"
    model_dir1 = out_dir + "/temp/" + "hgt_parameter/"
    model_dir2 = out_dir + "/temp/" + "AE_parameter/"
    model_dir3 = out_dir + "/temp/" + "AE_loss/"
    model_dir4 = out_dir + "/temp/" + "hgt_loss/"
    model_dir5 = out_dir + "/temp/" + "roc_point/"
    model_dir6 = out_dir + "/temp/" + "test_index/"
    if os.path.exists(path1) is False:
        os.mkdir(out_dir + "/temp")
    if os.path.exists(model_dir1) is False:
        os.mkdir(out_dir + "/temp" + "/hgt_parameter")
    if os.path.exists(model_dir2) is False:
        os.mkdir(out_dir + "/temp" + "/AE_parameter")
    if os.path.exists(model_dir3) is False:
        os.mkdir(out_dir + "/temp" + "/AE_loss")
    if os.path.exists(model_dir4) is False:
        os.mkdir(out_dir + "/temp" + "/hgt_loss")
    if os.path.exists(model_dir5) is False:
        os.mkdir(out_dir + "/temp" + "/roc_point")
    if os.path.exists(model_dir6) is False:
        os.mkdir(out_dir + "/temp" + "/test_index")

    # load data
    LOGGER.info(
        "Loading species abundance matrix, sample labels, phylogenetics and metabolic relationships to model"
    )

    (
        gene_cell_matrix1,
        gene_cell_matrix,
        cell_label,
        gene_cell,
        gene_name,
        cell_name,
    ) = load_data(args.input1, args.input2, sep=",", col_name=True, row_name=True)
    Label_transform = LabelEncoder()
    Label_transform.fit(cell_label)
    cell_label_num = Label_transform.fit_transform(cell_label)
    num_type = len(Label_transform.classes_)
    gene_cell_matrix = gene_cell_matrix.astype("float")
    gene_cell = gene_cell.astype("float")
    cell_set = {int(k): [] for k in cell_label_num}
    cell_name_set = {int(k): [] for k in cell_label_num}
    for j, i in enumerate(cell_label_num):
        for k in cell_set:
            if int(i) == k:
                cell_set[k].append(gene_cell[:, j])
                cell_name_set[k].append(cell_label.index.values[j])
    weight = []
    for i in range(num_type):
        label_count = len(cell_set[i])
        weight.append(1 - (label_count / gene_cell_matrix.shape[1]))
    train_set = []
    train_label = []
    train_cell_name1 = []
    test_set = []
    test_label = []
    test_cell_name1 = []
    for i, k in cell_set.items():
        (
            train_cell1,
            test_cell1,
            train_cell1_name,
            test_cell1_name,
        ) = split_cell_train_test(args, k, cell_name_set[i])
        train_set.append(train_cell1)
        test_set.append(test_cell1)
        train_cell_name1.append(train_cell1_name)
        test_cell_name1.append(test_cell1_name)
        train_label.extend([i] * len(train_cell1))
        test_label.extend([i] * len(test_cell1))
    train_cell = []
    test_cell = []
    train_cell_name = []
    test_cell_name = []
    for i in train_set:
        for j in i:
            train_cell.append(j)
    for i in train_cell_name1:
        for j in i:
            train_cell_name.append(j)
    for i in test_set:
        for j in i:
            test_cell.append(j)
    for i in test_cell_name1:
        for j in i:
            test_cell_name.append(j)
    train_cell11, train_label11, train_cell_name = shuffle(
        train_cell, train_label, train_cell_name
    )
    test_cell, test_label, test_cell_name = shuffle(
        test_cell, test_label, test_cell_name
    )
    # over_num = Counter(np.array(train_label)).most_common(1)[0][1]
    oversampling = RandomOverSampler(sampling_strategy="not majority", random_state=0)
    train_cell1, train_label1 = oversampling.fit_resample(train_cell11, train_label11)
    # print(train_label11)
    # print(train_label1)
    train_over_name = list(1 for i in range(len(train_label1) - len(train_label11)))
    train_over_cell, train_over_label, train_over_name = shuffle(
        train_cell1[len(train_label11) : len(train_label1)],
        train_label1[len(train_label11) : len(train_label1)],
        train_over_name,
    )
    train_cell = train_cell11 + train_over_cell
    train_label = train_label11 + train_over_label
    # print(train_label)
    train_cell_name = train_cell_name + train_over_name
    train_cell = np.asarray(train_cell)
    test_cell = np.asarray(test_cell)
    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
    # print(test_cell.shape)
    # print(train_cell.shape)
    cuda = args.cuda  # 'cpu'#-1
    if cuda == -1:
        device = torch.device("cpu")
        LOGGER.info("Using CPU")
    else:
        device = torch.device("cuda:" + str(cuda))
        LOGGER.info(f"Using GPU device: {str(cuda)}")

    LOGGER.info("Autoencoder is trainning...")

    train_cell = train_cell.T
    test_cell = test_cell.T
    l1 = []
    l2 = []
    l3 = []
    if args.reduction == "AE":
        gene = torch.tensor(train_cell, dtype=torch.float32).to(device)

        ba = train_cell.shape[0]
        loader1 = torchdata.DataLoader(
            gene, ba
        )  # 这里为什么gene的loader1(365-397行)与cell的loader2(399-425行)不一样？？

        EPOCH_AE = 250
        model = AE(dim=train_cell.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_func = nn.MSELoss()
        for epoch in range(EPOCH_AE):
            embedding1 = []
            for _, batch_x in enumerate(loader1):

                decoded, encoded = model(batch_x)
                # encoded1 , decoded1 = Coder2(cell)
                loss = loss_func(batch_x, decoded)
                l1.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                embedding1.append(encoded)
            # print('Epoch :', epoch,'|','train_loss:%.12f'%loss.data)
        if (
            gene.shape[0] % ba != 0
        ):  # 这里的意思是如果gene.shape[0]中不是含有整数个ba(也就是batch_size)   %是取模的意思
            torch.stack(
                embedding1[0 : int(gene.shape[0] / ba)]
            )  # stack是指在维度上连接（concatenate）若干个张量(这些张量形状相同）
            a = torch.stack(embedding1[0 : int(gene.shape[0] / ba)])
            a = a.view(
                ba * int(gene.shape[0] / ba), args.in_dim
            )  # view()的作用相当于numpy中的reshape,重新定义矩阵的形状    ??是为了把a从三维(EPOCH_AE*ba*256)的张量转换为二维的张量？？
            encoded = torch.cat((a, encoded), 0)  # cat也是拼接的意思    ??a里面不就是encoded吗？？

        else:
            encode = torch.stack(embedding1)
            encoded = encode.view(gene.shape[0], args.in_dim)

        # if train_cell.shape[1]<5000:           #gene_cell[0]中是基因，而gene_cell[1]中是cell
        #    ba1 = train_cell.shape[1]
        # else:
        #    ba1 = 5000
        ba1 = train_cell.shape[1]
        cell = torch.tensor(train_cell.T, dtype=torch.float32).to(device)
        # if test_cell.shape[1]<5000:           #gene_cell[0]中是gene，而gene_cell[1]中是cell
        #    ba2 = test_cell.shape[1]
        # else:
        #    ba2 = 5000
        ba2 = test_cell.shape[1]
        cell1 = torch.tensor(test_cell.T, dtype=torch.float32).to(device)
        # 这里是将gene_cell矩阵转置,即使得每次进入AE中的都是矩阵中的行向量
        loader2 = torchdata.DataLoader(cell, ba1)
        loader3 = torchdata.DataLoader(cell1, ba2)
        model2 = AE(dim=train_cell.shape[0]).to(device)
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)  # ,weight_decay=1e-2)
        EPOCH_AE2 = 250
        for epoch in range(EPOCH_AE2):
            embedding1 = []
            embedding2 = []
            for _, batch_x in enumerate(loader2):
                decoded2, encoded2 = model2(batch_x)
                loss = loss_func(batch_x, decoded2)
                l2.append(loss.item())
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                embedding1.append(encoded2)
                print("Epoch :", epoch, "|", "train_loss:%.12f" % loss.data)
            for _, x in enumerate(loader3):
                decoded3, encoded3 = model2(x)
                test_loss = loss_func(x, decoded3)
                l3.append(test_loss.item())
                embedding2.append(encoded3)

        if cell.shape[0] % ba1 != 0:
            torch.stack(embedding1[0 : int(cell.shape[0] / ba1)])
            a = torch.stack(embedding1[0 : int(cell.shape[0] / ba1)])
            a = a.view(ba1 * int(cell.shape[0] / ba1), args.in_dim)
            encoded2 = torch.cat((a, encoded2), 0)
            # encoded2.shape
        else:
            encode = torch.stack(embedding1)
            encoded2 = encode.view(cell.shape[0], args.in_dim)
        encode2 = torch.stack(embedding2)
        encoded3 = encode2.view(cell1.shape[0], args.in_dim)

    plt.figure()
    plt.plot(l1, "r-")
    plt.title("species loss per iteration")
    plt.savefig(model_dir3 + "species_" + str(file0) + ".png")
    plt.figure()
    plt.plot(l2, "r-")
    plt.plot(l3, "g-")
    plt.title("train-test loss per iteration")
    plt.savefig(model_dir3 + "sample_" + str(file0) + ".png")
    if args.reduction == "raw":  # 这里应该是对于输入的矩阵不进行降维
        encoded = torch.tensor(gene_cell, dtype=torch.float32).to(device)
        encoded2 = torch.tensor(np.transpose(gene_cell), dtype=torch.float32).to(device)

    if os.path.exists(model_dir2) is False:
        os.mkdir(model_dir2)
    torch.save(model2.state_dict(), model_dir2 + file0)

    gene_name11 = [str(i) for i in gene_name.astype("int64")]
    gene_name12 = [int(i) for i in gene_name]
    species_species = pd.read_csv(metabolic_path, sep=",", index_col=0)
    species_species = species_species.loc[:, gene_name11]
    species_species = species_species.loc[gene_name12, :]
    species_name = species_species.columns.values
    species_matrix = species_species.values
    # species_matrix = np.zeros((1218,1218))
    g12 = np.nonzero(species_matrix)[0]
    c22 = np.nonzero(species_matrix)[1]
    edge12 = list(g12)
    edge22 = list(c22)

    species_species1 = pd.read_csv(phylo_path, sep=",", index_col=0)
    species_species1 = species_species1.loc[:, gene_name11]
    species_species1 = species_species1.loc[gene_name12, :]
    species_name1 = species_species1.columns.values
    species_matrix1 = species_species1.values
    # species_matrix = np.zeros((1218,1218))
    g13 = np.nonzero(species_matrix1)[0]
    c23 = np.nonzero(species_matrix1)[1]
    edge13 = list(g13)
    edge23 = list(c23)

    # target_nodes = np.arange(train_cell.shape[0]+train_cell.shape[1])            #np.shape(gene_cell)[0]为gene_cell行的长度 np.shape[1]为gene_cell列的长度
    # gene cell
    g11 = np.nonzero(train_cell)[0]  # np.nonzero[0]返回gene_cell中行的非零元的索引
    c21 = (
        np.nonzero(train_cell)[1] + train_cell.shape[0]
    )  # np.nonzero[1]返回gene_cell中列的非零元的索引
    edge11 = list(g11)
    edge21 = list(c21)
    # edge1 = edge11+edge12
    # edge2 = edge21+edge22
    # edge_index = torch.tensor([edge1, edge2], dtype=torch.long)
    x = {
        "gene": torch.tensor(encoded, dtype=torch.float),
        "cell": torch.tensor(encoded2, dtype=torch.float),
    }
    edge_index_dict = {
        ("gene", "g_c", "cell"): torch.tensor([g11, c21], dtype=torch.long),
        ("cell", "c_g", "gene"): torch.tensor([c21, g11], dtype=torch.long),
        ("gene", "g_g", "gene"): torch.tensor([g12, c22], dtype=torch.long),
        ("gene1", "g_g", "gene1"): torch.tensor([g13, c23], dtype=torch.long),
    }

    edge_reltype = {
        ("gene", "g_c", "cell"): torch.tensor([g11, c21]).shape[1],
        ("cell", "c_g", "gene"): torch.tensor([c21, g11]).shape[1],
        ("gene", "g_g", "gene"): torch.tensor([g12, c22]).shape[1],
        ("gene1", "g_g", "gene1"): torch.tensor([g13, c23]).shape[1],
    }
    num_nodes_dict = {"gene": train_cell.shape[0], "cell": train_cell.shape[1]}
    data = Data(
        edge_index_dict=edge_index_dict,
        edge_reltype=edge_reltype,
        num_nodes_dict=num_nodes_dict,
        x=x,
    )

    a = np.nonzero(train_cell)[0]
    b = np.nonzero(train_cell)[1]
    node_type = list(np.zeros(train_cell.shape[0])) + list(np.ones(train_cell.shape[1]))
    # node_type1 = pd.DataFrame(node_type)
    # node_type1.to_csv('/fs/ess/PCON0022/yuhan/HGT/result/check_repeat/'+'node_type'+str(file0)+'.csv', sep=",")
    node_type = torch.LongTensor(node_type)

    # node_type = node_type.to(device)
    node_feature = []
    for t in ["gene", "cell"]:
        if args.reduction != "raw":
            node_feature += list(x[t])
        else:
            node_feature[t_i] = torch.tensor(x[t], dtype=torch.float32).to(device)
    if args.reduction != "raw":
        node_feature = torch.stack(node_feature)
        node_feature = torch.tensor(node_feature, dtype=torch.float32)
        node_feature = node_feature.to(device)
    # node_feature1 = node_feature.detach().numpy()
    # process_encoded1 = pd.DataFrame(node_feature1)
    # process_encoded1.to_csv('/fs/ess/PCON0022/yuhan/HGT/result/check_repeat/'+'process_encoded'+str(file0)+'.csv', sep=",")
    # print(node_feature)
    edge_index1 = data["edge_index_dict"][("gene", "g_c", "cell")]
    edge_index2 = data["edge_index_dict"][("cell", "c_g", "gene")]
    edge_index3 = data["edge_index_dict"][("gene", "g_g", "gene")]
    edge_index4 = data["edge_index_dict"][("gene1", "g_g", "gene1")]
    edge_index = torch.cat((edge_index1, edge_index2, edge_index3, edge_index4), 1)
    # edge_index = torch.cat((edge_index1,edge_index2),1)
    edge_type = (
        list(np.zeros(len(edge_index1[1])))
        + list(np.ones(len(edge_index2[1])))
        + list(2 for i in range(len(edge_index3[1])))
        + list(3 for i in range(len(edge_index4[1])))
    )
    edge_time = torch.LongTensor(list(np.zeros(len(edge_index[1]))))
    edge_type = torch.LongTensor(edge_type)
    edge_index = torch.LongTensor(edge_index.numpy())

    test_g11 = np.nonzero(test_cell)[0]
    test_c21 = np.nonzero(test_cell)[1] + test_cell.shape[0]

    # test_edge1 = test_edge11+edge12
    # test_edge2 = test_edge21+edge22
    # test_edge_index = torch.tensor([test_edge1, test_edge2], dtype=torch.long)
    test_x = {
        "gene": torch.tensor(encoded, dtype=torch.float),
        "cell": torch.tensor(encoded3, dtype=torch.float),
    }  # batch of gene all cells
    # edge_index_dict2 = {('gene','g_c','cell'): torch.tensor([g11, c21], dtype=torch.long)}
    edge_index_dict2 = {
        ("gene", "g_c", "cell"): torch.tensor([test_g11, test_c21], dtype=torch.long),
        ("cell", "c_g", "gene"): torch.tensor([test_c21, test_g11], dtype=torch.long),
        ("gene", "g_g", "gene"): torch.tensor([g12, c22], dtype=torch.long),
        ("gene1", "g_g", "gene1"): torch.tensor([g13, c23], dtype=torch.long),
    }

    edge_reltype2 = {
        ("gene", "g_c", "cell"): torch.tensor([test_g11, test_c21]).shape[1],
        ("cell", "c_g", "gene"): torch.tensor([test_c21, test_g11]).shape[1],
        ("gene", "g_g", "gene"): torch.tensor([g12, c22]).shape[1],
        ("gene1", "g_g", "gene1"): torch.tensor([g13, c23]).shape[1],
    }

    num_nodes_dict2 = {"gene": test_cell.shape[0], "cell": test_cell.shape[1]}
    data2 = Data(
        edge_index_dict=edge_index_dict2,
        edge_reltype=edge_reltype2,
        num_nodes_dict=num_nodes_dict2,
        x=test_x,
    )
    # a = np.nonzero(adj)[0]
    # b = np.nonzero(adj)[1]
    node_type8 = list(np.zeros(test_cell.shape[0])) + list(np.ones(test_cell.shape[1]))
    node_type8 = torch.LongTensor(node_type8)
    # node_type8 = node_type8.to(device)
    node_feature2 = []
    for t in ["gene", "cell"]:
        if args.reduction != "raw":
            node_feature2 += list(test_x[t])
        else:
            node_feature2[t_i] = torch.tensor(test_x[t], dtype=torch.float32).to(device)
    if args.reduction != "raw":
        node_feature2 = torch.stack(node_feature2)
        node_feature2 = torch.tensor(node_feature2, dtype=torch.float32)
        node_feature2 = node_feature2.to(device)
    test_edge_index1 = data2["edge_index_dict"][("gene", "g_c", "cell")]
    test_edge_index2 = data2["edge_index_dict"][("cell", "c_g", "gene")]
    test_edge_index3 = data2["edge_index_dict"][("gene", "g_g", "gene")]
    test_edge_index4 = data2["edge_index_dict"][("gene1", "g_g", "gene1")]
    test_edge_index = torch.cat(
        (test_edge_index1, test_edge_index2, test_edge_index3, test_edge_index4), 1
    )
    # test_edge_index = torch.cat((test_edge_index1,test_edge_index2),1)
    edge_type2 = (
        list(np.zeros(len(test_edge_index1[1])))
        + list(np.ones(len(test_edge_index2[1])))
        + list(2 for i in range(len(test_edge_index3[1])))
        + list(3 for i in range(len(test_edge_index4[1])))
    )
    edge_time2 = torch.LongTensor(list(np.zeros(len(test_edge_index[1]))))
    edge_type2 = torch.LongTensor(edge_type2)
    test_edge_index = torch.LongTensor(test_edge_index.numpy())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # debuginfoStr('Cell Graph constructed and pruned')
    if args.reduction != "raw":
        gnn = GNN(
            conv_name=args.layer_type,
            in_dim=encoded.shape[1],
            n_hid=args.n_hid,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            num_types=2,
            num_relations=4,
            use_RTE=False,
        ).to(device)
    else:
        gnn = GNN_from_raw(
            conv_name=args.layer_type,
            in_dim=[encoded.shape[1], encoded2.shape[1]],
            n_hid=args.n_hid,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            num_types=2,
            num_relations=4,
            use_RTE=False,
            AEtype=args.AEtype,
        ).to(device)
    classifier = Classifier(args.n_hid, num_type).to(device)

    args_optimizer = "adamw"
    if args_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [{"params": gnn.parameters()}, {"params": classifier.parameters()}],
            lr=args.lr,
        )
    elif args_optimizer == "adam":
        optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    elif args_optimizer == "sgd":
        optimizer = torch.optim.SGD(gnn.parameters(), lr=args.lr)
    elif args_optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(gnn.parameters(), lr=args.lr)
    # gnn.double()

    # model, optimizer = amp.initialize(gnn, optimizer, opt_level="O1")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=5, verbose=True
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # index = np.argwhere(train_label<5)
    # train_label1 = np.delete(train_label,index)
    train_label1 = torch.LongTensor(train_label.flatten()).to(device)  ##一会加个1
    test_label1 = torch.LongTensor(test_label.flatten()).to(device)

    loss_function = focal_loss(alpha=weight, gamma=args.gamma, num_classes=num_type).to(
        device
    )
    # k=[]
    train_loss_all = []
    train_F1 = []
    test_loss_all = []
    test_F1 = []
    for epoch in np.arange(args.epoch):
        gnn.train()
        classifier.train()
        L = 0
        if args.reduction == "raw":
            node_rep, node_decoded_embedding = gnn.forward(
                node_feature,
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
        else:
            node_rep = gnn.forward(
                node_feature,
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
        train_att1 = gnn.att1
        train_att2 = gnn.att2
        if args.rep == "T":  # 为了结果可复现
            node_rep = torch.trunc(node_rep * 10000000000) / 10000000000
            if args.reduction == "raw":
                for t in types:
                    t_i = node_dict[t][1]
                    node_decoded_embedding[t_i] = (
                        torch.trunc(node_decoded_embedding[t_i] * 10000000000)
                        / 10000000000
                    )

        gene_matrix = node_rep[
            node_type == 0,
        ]
        cell_matrix = node_rep[
            node_type == 1,
        ]
        decoder = torch.mm(gene_matrix, cell_matrix.t())
        adj = torch.tensor(train_cell, dtype=torch.float32).to(device)
        # adj1 = np.matmul(train_cell,train_cell.T)
        # adj1 = torch.tensor(adj1,dtype=torch.float32).to(device)
        KL_loss = F.kl_div(
            decoder.softmax(dim=-1).log(), adj.softmax(dim=-1), reduction="sum"
        )
        pre_label, pre_score = classifier.forward(
            cell_matrix
        )  # 到底应该在哪个循环之下？为什么是495，还差5个？？？下面定义交叉熵，还是在这个循环嵌套中并且用args_optimizer来优化
        cross_loss = loss_function(pre_score, pre_label, train_label1)
        loss = args.kl_coef * KL_loss + cross_loss
        train_loss_all.append(loss.item())
        true_score = label_binarize(train_label, classes=[i for i in range(num_type)])
        pre_score2 = pre_score.cpu().detach().numpy()
        train_pre_score = [np.argmax(i) for i in pre_score2]
        train_f1 = f1_score(train_label, train_pre_score, average="macro")
        train_F1.append(train_f1)

        L = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(L)
        print("Epoch :", epoch + 1, "|", "train_loss:%.12f" % (L))
        gnn.eval()
        classifier.eval()
        test_node_rep = gnn.forward(
            node_feature2,
            node_type8.to(device),
            edge_time2.to(device),
            test_edge_index.to(device),
            edge_type2.to(device),
        )
        test_att1 = gnn.att1
        test_att2 = gnn.att2
        if args.rep == "T":  # 为了结果可复现
            test_node_rep = torch.trunc(node_rep * 10000000000) / 10000000000
            if args.reduction == "raw":
                for t in types:
                    t_i = node_dict[t][1]
                    # print("t_i="+str(t_i))
                    node_decoded_embedding[t_i] = (
                        torch.trunc(node_decoded_embedding[t_i] * 10000000000)
                        / 10000000000
                    )

        test_gene_matrix = test_node_rep[
            node_type8 == 0,
        ]
        test_cell_matrix = test_node_rep[
            node_type8 == 1,
        ]
        test_decoder = torch.mm(test_gene_matrix, test_cell_matrix.t())
        test_adj = torch.tensor(test_cell, dtype=torch.float32).to(device)
        test_KL_loss = F.kl_div(
            test_decoder.softmax(dim=-1).log(),
            test_adj.softmax(dim=-1),
            reduction="sum",
        )
        test_pre_label, test_pre_score = classifier.forward(
            test_cell_matrix
        )  # 到底应该在哪个循环之下？为什么是495，还差5个？？？下面定义交叉熵，还是在这个循环嵌套中并且用args_optimizer来优化
        test_cross_loss = loss_function(test_pre_score, test_pre_label, test_label1)
        test_loss = args.kl_coef * test_KL_loss + test_cross_loss
        test_loss_all.append(test_loss.item())
        pre_score1 = test_pre_score.cpu().detach().numpy()
        pre_score11 = [np.argmax(i) for i in pre_score1]
        test_f1 = f1_score(test_label, pre_score11, average="macro")
        test_F1.append(test_f1)
        print("Epoch :", epoch + 1, "|", "test_loss:%.12f" % (test_loss.item()))

    attention1 = []
    attention1_no_softmax = []
    attention1.append(train_att1[: len(np.array(edge_index1[0])), :])
    attention1_no_softmax.append(train_att2[: len(np.array(edge_index1[0])), :])
    attention1 = attention1[0].cpu().detach().numpy()
    attention1_no_softmax = attention1_no_softmax[0].cpu().detach().numpy()
    gene_name1 = list(gene_name)
    edge_index1 = torch.LongTensor(edge_index1).numpy()
    gene_name1 = [gene_name1[i] for i in list(np.array(edge_index1[0]))]
    cell_name1 = [
        train_cell_name[i] for i in list(np.array(edge_index1[1] - train_cell.shape[0]))
    ]
    label_name1 = [
        train_label[i] for i in list(np.array(edge_index1[1] - train_cell.shape[0]))
    ]
    label_name1 = Label_transform.inverse_transform(label_name1)
    attention2 = []
    attention2_no_softmax = []
    gene_name2 = list(gene_name)
    gene_name2 = [gene_name2[i] for i in list(np.array(test_edge_index1[0]))]
    test_cell_name = list(test_cell_name)
    cell_name2 = [
        test_cell_name[i]
        for i in list(np.array(test_edge_index1[1] - test_cell.shape[0]))
    ]
    test_label = list(test_label)
    label_name2 = [
        test_label[i] for i in list(np.array(test_edge_index1[1] - test_cell.shape[0]))
    ]
    label_name2 = Label_transform.inverse_transform(label_name2)
    attention2.append(test_att1[: len(np.array(test_edge_index1[0])), :])
    attention2_no_softmax.append(test_att2[: len(np.array(test_edge_index1[0])), :])
    attention2 = attention2[0].cpu().detach().numpy()
    attention2_no_softmax = attention2_no_softmax[0].cpu().detach().numpy()

    plt.figure()
    plt.plot(train_loss_all, "r-")
    plt.plot(test_loss_all, "g-")
    plt.title("train-test loss per iteration")
    plt.savefig(model_dir4 + "loss_" + str(file0) + ".png")
    plt.figure()
    plt.plot(train_F1, "r-")
    plt.plot(test_F1, "g-")
    plt.title("train-test F1 per iteration")
    plt.savefig(model_dir4 + "F1_" + str(file0) + ".png")
    n_classes = true_score.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # pre_score = pre_score.detach().numpy()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(true_score[:, i], pre_score2[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="b",
        linestyle=":",
        linewidth=4,
    )
    colors = ["m", "c", "r", "g", "y"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multi-class")
    plt.legend(loc="lower right")
    plt.savefig(model_dir5 + str(file0) + ".png")

    if os.path.exists(model_dir1) == False:
        os.mkdir(model_dir1)
    state1 = {
        "model_1": gnn.state_dict(),
        "model_2": classifier.state_dict(),
        "optimizer": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state1, model_dir1 + file0)

    # pre_score1 = [np.argmax(i) for i in pre_score1]
    target_names = Label_transform.inverse_transform([i for i in range(num_type)])
    others = classification_report(
        y_true=test_label,
        y_pred=pre_score11,
        labels=[i for i in range(num_type)],
        target_names=target_names,
        output_dict=True,
    )
    others = pd.DataFrame(others).transpose()
    others.to_csv(model_dir6 + str(file0) + ".csv", index=0)

    gene_name = gene_name1 + gene_name2
    cell_name = cell_name1 + cell_name2
    gene_name = np.array(gene_name)
    cell_name = np.array(cell_name)
    over_index = np.argwhere(cell_name != "1").flatten()
    cell_name = cell_name[over_index]
    gene_name = gene_name[over_index]
    label_name1 = list(label_name1)
    label_name2 = list(label_name2)
    label_name = label_name1 + label_name2
    label_name = np.array(label_name)
    label_name = label_name[over_index]
    attention = np.concatenate([attention1, attention2], axis=0)
    attention = attention[over_index]
    attention_no_softmax = np.concatenate(
        [attention1_no_softmax, attention2_no_softmax], axis=0
    )
    attention_no_softmax = attention_no_softmax[over_index]
    # g = np.nonzero(adj)[0]
    # c = np.nonzero(adj)[1]+adj.shape[0]
    name1 = pd.DataFrame(gene_name, columns=["taxa_id"])
    name2 = pd.DataFrame(cell_name, columns=["Sample"])
    name3 = pd.DataFrame(label_name, columns=["cancer_type"])
    df = pd.DataFrame(
        attention,
        columns=[
            "attention_head_1",
            "attention_head_2",
            "attention_head_3",
            "attention_head_4",
            "attention_head_5",
            "attention_head_6",
            "attention_head_7",
            "attention_head_8",
        ],
    )
    df2 = pd.DataFrame(
        attention_no_softmax,
        columns=[
            "attention_head_1",
            "attention_head_2",
            "attention_head_3",
            "attention_head_4",
            "attention_head_5",
            "attention_head_6",
            "attention_head_7",
            "attention_head_8",
        ],
    )
    df = pd.concat([name1, name2, name3, df], axis=1)
    df2 = pd.concat([name1, name2, name3, df2], axis=1)
    df.to_csv(att_file1, sep=",", index=True)
    LOGGER.info(f"The final attention score (after softmax) is saved in {att_file1}")


def create_argument_parser():
    LOGGER.info(
        f"pyMEGA: A deep learning package for identifying cancer-associated tissue-resident. Version: {VERSION}"
    )
    parser = argparse.ArgumentParser(
        description="Training GNN on species_sample graph",
    )

    parser.add_argument(
        "-o",
        default="./",
        help="The output directory. default: current working directory.",
    )
    parser.add_argument(
        "-epoch",
        type=int,
        default=30,
        help="Number of training iteration. default: 30",
    )

    parser.add_argument(
        "-input1", default=None, help="The absolute path of abundance matrix."
    )
    parser.add_argument(
        "-input2",
        default=None,
        help="The absolute path of metadata of the abundance matrix.",
    )
    parser.add_argument(
        "-db", default=None, help="The absolute path of Gut metebolic database."
    )
    # Feature extration
    parser.add_argument(
        "-num", type=float, default=0.9, help="the num of training data. default: 0.9"
    )
    parser.add_argument(
        "-reduction",
        type=str,
        default="AE",
        help="the method for feature extraction, pca, raw. default: AE",
    )

    parser.add_argument(
        "-in_dim",
        type=int,
        default=256,
        help="Number of hidden dimension (AE) default: 256",
    )

    # GAE
    parser.add_argument(
        "-kl_coef",
        type=float,
        default=0.00005,  # KL co-efficient
        help="coefficient of regular term. default: 0.00005",
    )
    parser.add_argument(
        "-gamma",
        type=float,
        default=2.5,
        help="coefficient of focal loss. default: 2.5",
    )
    parser.add_argument(
        "-lr", type=float, default=0.003, help="learning rate. default: 0.003"
    )
    parser.add_argument(
        "-n_hid", type=int, default=128, help="Number of hidden dimension. default: 128"
    )
    parser.add_argument(
        "-n_heads", type=int, default=8, help="Number of attention head. default: 8"
    )
    parser.add_argument(
        "-n_layers", type=int, default=2, help="Number of GNN layers. default: 2"
    )
    parser.add_argument(
        "-dropout", type=float, default=0, help="Dropout ratio. default: 0"
    )
    parser.add_argument(
        "-layer_type", type=str, default="hgt", help="the layer type for GAE"
    )
    parser.add_argument(
        "-loss", type=str, default="cross", help="the loss for GAE. default: cross"
    )

    parser.add_argument(
        "-cuda",
        type=int,
        default=1,
        help="the GPU device number to use. Set cuda=-1 to use cpu; cuda=0 to use GPU0. default: 0",
    )

    parser.add_argument(
        "-rep", type=str, default="iT", help="precision truncation. default: iT"
    )

    parser.add_argument(
        "-AEtype",
        type=int,
        default=1,
        help="AEtype1: embedding node autoencoder. 2:HGT node autoencode. default: 1",
    )

    parser.add_argument(
        "-seed",
        type=int,
        required=False,
        default=0,
        help="Seed value for regressor random state initialization. default: 0",
    )
    # the weight of each cancer type in loss function

    # parser.add_argument('--weight', type = float, nargs = '+', default=[0.755, 0.912, 0.696, 0.882, 0.755],
    #                   help='weight of each class in loss function')

    return parser


def set_seed(args):
    seed = args.seed
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


def create_out_dir(dir):
    out_dir = os.path.abspath(dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        LOGGER.info(f"Directory '{out_dir}' created successfully!")

    LOGGER.info(f"Output files will be save to: {out_dir}")
    return out_dir


def result_selection(att_path, abundance_path, out_dir, base_filename, t, pv):
    f = open(att_path)
    p = pd.read_csv(f, sep=",")

    df = pd.read_csv(
        abundance_path, sep=",", index_col=0
    )  # obtain the total number of species and obtain the output path

    file1 = out_dir + "/" + base_filename + "_taxa_num.csv"
    file2 = out_dir + "/" + base_filename + "_final_taxa.txt"

    # obtain the dictionary consisting of diseases and corresponding samples
    disease_list = list(set(p["cancer_type"].tolist()))
    value = []
    # print (disease_list[1])
    for i in range(len(disease_list)):
        sample_lis = []
        for j in range(p.shape[0]):
            if p.iloc[j, 3] == disease_list[i]:
                sample_lis.append(p.iloc[j, 2])
        sample_list = list(set(sample_lis))
        value.append(sample_list)
    dictionary = dict(zip(disease_list, value))

    # for each cancer type: key ;  obtain a dict: dict{cancer_type: [{taxa of each sample}];}
    total_lis_taxa = []
    for key, value in dictionary.items():
        # print (key)
        lis_taxa = []
        for k in range(len(value)):
            # for each sample value[k],
            tem_p = p.loc[p["Sample"] == value[k]]
            taxa_ = set()
            for j in range(4, 12):
                # print (tem_p.iloc[:,j])
                a = np.array(tem_p.iloc[:, j])
                # print (a)
                lower_q = np.quantile(a, 0.25, interpolation="lower")
                higher_q = np.quantile(a, 0.75, interpolation="higher")
                q1_q3 = list(
                    tem_p.iloc[:, j][
                        (tem_p.iloc[:, j] > lower_q) & (tem_p.iloc[:, j] < higher_q)
                    ]
                )
                mean_ = np.mean(q1_q3)
                std_ = np.std(q1_q3)
                # caculate threshold
                thre_ = mean_ + std_ * t
                taxa_j = set(tem_p["taxa_id"][tem_p.iloc[:, j] > thre_])
                taxa_ = taxa_.union(taxa_j)
            lis_taxa.append(taxa_)
        total_lis_taxa.append(lis_taxa)
    dictionary_taxa = dict(zip(disease_list, total_lis_taxa))

    # calculate the number of each taxa: dict_taxa_num_all; the selected taxa number of each sample: dict_sample_taxa_num_all
    dict_taxa_num_t = []
    sample_taxa_num_t = []
    for key, value in dictionary_taxa.items():
        # print (key)
        # print (len(value))
        set_union = set()
        sample_taxa_num = []
        for i in range(len(value)):
            set_union = set_union.union(value[i])
            sample_taxa_num.append(len(value[i]))
        sample_taxa_num_t.append(sample_taxa_num)
        list_union = list(set_union)

        # caculate the number of each taxa for each phenotype
        list_num = []
        for k in range(len(list_union)):
            s = 0
            for j in range(len(value)):
                s = s + int(list_union[k] in value[j])
                # print (int(list_union[k] in value[j]))
            list_num.append(s)
        dict_taxa_num = dict(zip(list_union, list_num))
        dict_taxa_num_t.append(dict_taxa_num)

    dict_taxa_num_all = dict(zip(disease_list, dict_taxa_num_t))
    dict_sample_taxa_num_all = dict(zip(disease_list, sample_taxa_num_t))
    # print (dict_taxa_num_all)
    # print (dict_sample_taxa_num_all)

    df1 = pd.DataFrame(dict_taxa_num_all).fillna(0)
    df1.to_csv(file1)

    # caculate the threshold for each phenotype: dict_thre = {phenotype:taxa_number_threshold}
    list_thre = []
    for key, value in dict_sample_taxa_num_all.items():
        a_max = max(value)
        b_min = min(filter(lambda x: x > 0, value))

        n = df.shape[0]
        m = a_max
        a = math.factorial(n) // (math.factorial(m) * math.factorial(n - m))
        n_1 = df.shape[0] - 1
        m_1 = a_max - 1
        b = math.factorial(n_1) // (math.factorial(m_1) * math.factorial(n_1 - m_1))
        rate = b / a
        m = m_1 = b_min
        a = math.factorial(n) // (math.factorial(m) * math.factorial(n - m))
        b = math.factorial(n_1) // (math.factorial(m_1) * math.factorial(n_1 - m_1))
        rate_1 = b / a

        p_value = 0
        for i in range(len(value), -1, -1):
            p_value = p_value + math.factorial(len(value)) // (
                math.factorial(i) * math.factorial(len(value) - i)
            ) * math.pow(rate, i) * math.pow(rate_1, len(value) - i)
            if p_value > pv:
                break
        list_thre.append(i)
    dict_thre = dict(zip(disease_list, list_thre))
    # print("The threshold of supported samples:", dict_thre.items())
    # print (dict_thre)

    # select the significant taxa for each phenotype
    list_final = []
    for key, value in dict_taxa_num_all.items():
        list_final_1 = []
        for key_1, value_1 in value.items():
            if value_1 > dict_thre[key]:
                list_final_1.append(key_1)
        list_final.append(list_final_1)
        # print("The number of selected taxa of ", key, ": ", len(list_final_1))
    dict_final_taxa = dict(zip(disease_list, list_final))

    # print final selected taxa into a file; as well as the number of supported samples
    f = open(file2, "w")
    for header, elem in dict_final_taxa.items():
        f.write(str(header) + "\t")
        for j in range(len(elem)):
            f.write(str(elem[j]))
            f.write("\t")
        f.write("\n")
    f.close()
    LOGGER.info(f"The number of samples with the selected taxa are saved in {file1}")
    LOGGER.info(f"The final selected taxa of each phenotype are saved in {file2}")


def main(argv=None):
    start_time = time.time()
    # Parse arguments.
    parser = create_argument_parser()
    args = parser.parse_args(args=argv)

    LOGGER.info(f"Your settings: {args}")
    out_dir = create_out_dir(args.o)
    base_filename = (args.input1).split("/")[-1].split(".")[0]

    # Preprocess
    species_list_path = get_metabolic_matrix(args.input1, out_dir)
    result = subprocess.run(
        ["Rscript", "r_scripts/taxize.r", species_list_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        LOGGER.info(
            "Successfully extracted NCBI taxonomy infomration using taxizedb package in R"
        )
    else:
        LOGGER.error(f"{result.stderr}")
    get_phylo_matrix(
        f"{out_dir}/{base_filename}_phy_relation.csv", args.input1, out_dir
    )

    # Run Micah
    micah(args, out_dir, base_filename)

    # final species selection
    result_selection(
        f"{out_dir}/{base_filename}_attention.csv",
        args.input1,
        out_dir,
        base_filename,
        args.t,
        args.pv,
    )

    # print running time
    end_time = time.time()
    total_time = end_time - start_time
    LOGGER.info(f"Total running time: {total_time} seconds")


if __name__ == "__main__":
    main()
