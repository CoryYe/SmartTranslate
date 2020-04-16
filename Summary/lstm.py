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
    def __init__(self, input_size, hidden_size, num_layers, batchSize, bidirectional,device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional= bidirectional
        self.num_layers = num_layers
        #self.dropout = dropout
        self.device = device
        self.batchSize = batchSize
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,bidirectional = self.bidirectional, dropout = 0.1)
        
        
    def forward(self, x, hidden):
        #print(x.shape)
        embed = self.embedding(x).view(len(x), self.batchSize, -1)
        output, (hiddens) = self.lstm(embed, (hidden))
        return output, hiddens
    
    def initHidden(self, batchS):
        return (torch.zeros(self.num_layers+1+int(self.bidirectional), batchS, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers+1+int(self.bidirectional), batchS, self.hidden_size, device=self.device))



class Attn(nn.Module):
    def __init__(self, hidden_size,device,batchS):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.batchS = batchS
        self.attn = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, hidden, encoder_outputs):
        length = len(encoder_outputs)
        attn_energies = torch.zeros(length, self.batchS).to(self.device)
        
        for i in range(length):
            #print(attn_energies.shape)
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
            
        return self.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
        
        
    def score(self, hidden, encoder_outputs):
        
        energy = self.attn(encoder_outputs)
        #print(hidden.shape)
        #print(energy.shape)
        energy = torch.bmm(hidden.view(self.batchS,1,-1), energy.view(self.batchS,-1,1))
        #print(energy.squeeze(2).shape)
        return energy.squeeze(2).squeeze(1)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout,batchSize,device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.batchSize = batchSize
        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.attn = nn.Linear(self.hidden_size *2, 182)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.LSTM = nn.LSTM(self.hidden_size * 3, self.hidden_size *2, num_layers = self.num_layers, bidirectional = False, dropout = dropout)
        self.out = nn.Linear(self.hidden_size * 4, self.output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.concat = nn.Linear(self.hidden_size*2,hidden_size)
        self.attn = Attn(self.hidden_size, self.device,self.batchSize)
        
    def initHidden(self, batchS):
        return (torch.zeros(self.num_layers, batchS, self.output_size),
                torch.zeros(self.num_layers, batchS, self.output_size))
    
    def forward(self, hidden, context, encoder_outputs, inputs):
        #print("pee")
        #print(inputs.shape)
        embedded = self.embedding(inputs).view(1, self.batchSize, -1)
        embedded = self.dropout(embedded)
        #print(embedded.shape)
        #print(context.shape)
        ri = torch.cat((embedded, context.unsqueeze(0)),2)
        
        #p#rint(ri.shape)
        ##print(hidden[0].shape)
        hids1 = hidden[0].view(self.num_layers, self.batchSize, -1)
        hids2 = hidden[1].view(self.num_layers, self.batchSize, -1)
        ro, (hidden) = self.LSTM(ri, (hids1,hids2))
        #weights = []
        #hiddens = hidden[0].view(1,batchS,-1)
        attn_weights = self.attn(ro.squeeze(0),encoder_outputs)

        attn_weights = attn_weights.squeeze(0).transpose(0,2).transpose(1,2)
        #print("antehuunaetd")
        #print(attn_weights.shape)
        #print(encoder_outputs.transpose(0,1).shape)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        #print(context.shape)
        ro = ro.squeeze(0)
        context = context.squeeze(1)
        output = self.softmax(self.out(torch.cat((ro,context),1)))

        return output, context, hidden, attn_weights
    
