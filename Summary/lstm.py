#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:35:38 2020

@author: cory
"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batchSize, bidirectional):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional= bidirectional
        self.num_layers = num_layers
        #self.dropout = dropout
        self.batchSize = batchSize
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,bidirectional = self.bidirectional)
        
        
    def forward(self, x, hidden, batchS):
        #print(x.shape)
        embed = self.embedding(x).view(1, batchS, -1)
        output, hiddens = self.lstm(embed, (hidden))
        return output, hiddens
    
    def initHidden(self, batchS):
        return (torch.zeros(self.num_layers+int(self.bidirectional), batchS, self.hidden_size),
                torch.zeros(self.num_layers+int(self.bidirectional), batchS, self.hidden_size))





class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout,batchSize):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchSize = batchSize
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size *2, 80)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size , num_layers = self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def initHidden(self, batchS):
        return (torch.zeros(self.num_layers, batchS, self.output_size),
                torch.zeros(self.num_layers, batchS, self.output_size))
    
    def forward(self, hidden, encoder_outputs, inputs, batchS):
        #print(inputs.shape)
        
        embedded = self.embedding(inputs).view(1, batchS, -1)
        embedded = self.dropout(embedded)
        
        #weights = []
        hiddens = hidden[0].view(1,batchS,-1)
        #print(embedded.shape)
        #print(hidden[0].shape)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hiddens[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, (hidden) = self.LSTM(output, (hidden[0].view(1,batchS,-1),hidden[1].view(1,batchS,-1)))

        output = self.softmax(self.out(output[0])).squeeze(1)
        
        return output, hidden, attn_weights
    