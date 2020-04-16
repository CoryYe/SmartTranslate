#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:22:36 2020

@author: cory
"""

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
import matplotlib as plt
import pandas as pd
import random 
import os


def main():
    
    #PATH = 'cnn/stories/cnn_dataset.pkl'
    #model = torch.load(PATH)
    #print(model[1])
    
    train = pd.concat([pd.read_csv(f'sumdata_1/train/train.article.txt', sep="\n"), 
                  pd.read_csv(f'sumdata_1/train/train.title.txt', sep="\n")], axis=1)
    train.columns = ["article", "title"]
    
    val = pd.concat([pd.read_csv(f'sumdata_1/train/valid.article.filter.txt', sep="\n"), 
                  pd.read_csv(f'sumdata_1/train/valid.title.filter.txt', sep="\n")], axis=1)
    val.columns = ["article", "title"]
    
    for index, row in train.iterrows():
        row['article'] = row['article'].replace(" ,", "")
        row['article'] = row['article'].replace("'", "")
        row['article'] = row['article'].replace(" ``", "")
        row['article'] = row['article'].replace(" -rrb-", "")
        row['article'] = row['article'].replace(" -lrb-", "")
        row['article'] = row['article'].replace(" <unk>", "")
    for index, row in val.iterrows():
        row['article'] = row['article'].replace(" ,", "")
        row['article'] = row['article'].replace("'", "")
        row['article'] = row['article'].replace(" ``", "")
        row['article'] = row['article'].replace(" -rrb-", "")
        row['article'] = row['article'].replace(" -lrb-", "")
        row['article'] = row['article'].replace(" <unk>", "")
    sample_train = train.sample(32000)
    sample_val = val.sample(4032)
    print(sample_train.iloc[1])
    #print(sample_val(1))
    
    sample_train.to_csv(f'train_sam.csv', index=None)
    sample_val.to_csv(f'valid_sam.csv', index=None)
    
if __name__ == "__main__":
    main()