#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:07:11 2020

@author: cory
"""
# suppress futurewarning of compatibility of numpy and tensorboard
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from torchtext import data
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random 
import spacy
import os
from lstm import Encoder, Decoder 




def word_ids_to_sentence(id_tensor, vocab, join=None):
    batch = [vocab.itos[ind] for ind in id_tensor] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)
    
    

def main():
    epochs = 10
    batchSize = 64
    lr = 0.00001
    
    
    #writer = SummaryWriter('./logs')
    #train = pd.read_csv(f'train_sam.csv')
    #train.columns = ["article", "title"]
    TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), lower=True, eos_token='_eos_')
    trn_data_fields = [("original", TEXT),
                   ("summary", TEXT)]
    train, valid = data.TabularDataset.splits(path=f'',
                                     train='train_sam.csv', validation='valid_sam.csv',
                                     format='csv', skip_header=True, fields=trn_data_fields)
    
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=200))
    
    train_iter, val_iter = data.BucketIterator.splits((train, valid),batch_sizes=(batchSize, batchSize),sort_key=lambda x: len(x.original), sort_within_batch=False, repeat=False)

    input_size = len(TEXT.vocab)
    hidden_size = 128*2
    dropout = 0.5
    num_layers = 1
    
    
    bidirectional = True
    
    encoder = Encoder(input_size, hidden_size, num_layers, batchSize,bidirectional)
    decoder = Decoder(input_size, num_layers*hidden_size*(1+int(bidirectional)), num_layers, dropout, batchSize)
    
    # define your LSTM loss function here
    #loss_func = F.cross_entropy()

    # define optimizer for lstm model
    #optim = Adam(model.parameters(), lr=lr)
    encoder_optimizer = Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = Adam(decoder.parameters(), lr=lr)
    losses = []
    valLosses = []
    originals = []
    summaries = []
    
    
    criterion = nn.NLLLoss()
    
    for epoch in range(epochs):
        originals = []
        genSumms = []
        origSumms= []
        batchNum = 0
        step = 0
        valBatchNum = 0
        for batch in train_iter:
            batchS = len(batch)
            batchNum+=1
            loss = 0
            orig = batch.original
            summ = batch.summary
            
            encoder_outputs = torch.zeros(80, encoder.hidden_size *(1+int(bidirectional)))
            encoder_hidden = encoder.initHidden(batchS)
            for ei in range(len(orig)):
                
                encoder_output, encoder_hidden = encoder.forward(
                        orig[ei], encoder_hidden,batchS)
                #print(encoder_outputs[ei].shape)
                #print(encoder_output[0,0].shape)
                encoder_outputs[ei] = encoder_output[0, 0]
            
            decoder_hidden = encoder_hidden
            decoder_input = torch.zeros(batchS).long()
            
            genSumm = []
            origSumm = []
            for di in range(len(summ)):
                decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_hidden,
                        encoder_outputs, decoder_input,batchS)
                loss += criterion(decoder_output, summ[di])
                #print(decoder_output)
                #print(summ[di])
                decoder_input = summ[di]  
                DO = decoder_output.detach().numpy()
                genSumm.append(np.argmax(DO[5]))
                origSumm.append(summ[di][5])
                #print(np.argmax(DO[5]))
                
                lossAvg = loss.item()/len(summ)
                

            encoder_optimizer.zero_grad()
            
            decoder_optimizer.zero_grad()
            
            loss.backward()
            
            #writer.add_scalar('training loss', loss.item(), step+1)
            step +=1
            
            encoder_optimizer.step()
            decoder_optimizer.step()
            genSumms.append(genSumm)
            origSumms.append(origSumm)
            
            originals.append(orig[:,5])
            genTensorO = torch.IntTensor(origSumms[0])
            genTensor = torch.IntTensor(genSumms[0])

            if (batchNum % 25 == 0):
                losses.append(lossAvg)
                print("Epoch: [{}/{}], Batch:[{}/{}], Loss: {}".format(
                        epoch, epochs, batchNum, len(train_iter),lossAvg))
                
                translatedOrig = word_ids_to_sentence(originals[0],TEXT.vocab,join = ' ')
                print(translatedOrig)
                translatedSummO = word_ids_to_sentence(genTensorO,TEXT.vocab,join = ' ')
                print(translatedSummO)
                translatedSumm = word_ids_to_sentence(genTensor,TEXT.vocab,join = ' ')
                print(translatedSumm)
                
                
            #genSumms = []
            
            if (batchNum % 25 == 0):
                for batchVal in val_iter:
                    valBatchNum+=1
                    valLoss = 0
                    batchS = len(batch)
                    valOrig = batchVal.original
                    valSumm = batchVal.summary
                    encoder_outputs = torch.zeros(80, encoder.hidden_size *(1+int(bidirectional)))
                    encoder_hidden = encoder.initHidden(batchS)
                    for ei in range(len(valOrig)):
                        encoder_output, encoder_hidden = encoder.forward(
                            valOrig[ei], encoder_hidden,batchS)
                        encoder_outputs[ei] = encoder_output[0, 0]
                        
                    decoder_hidden = encoder_hidden
                    decoder_input = torch.zeros(batchS).long()
                        
                    #genSumm = []
                    for di in range(len(valSumm)):
                        decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_hidden,
                            encoder_outputs, decoder_input,batchS)
                        valLoss += criterion(decoder_output, valSumm[di])
                        decoder_input = valSumm[di]
                        #DO = decoder_output.detach().numpy()
                        #genSumm.append(np.argmax(DO[5]))
                            
                    valLossAvg = valLoss.item()/len(valSumm)
                    
                    valLosses.append(valLossAvg)
                    print("VALLoss: {}".format(valLossAvg))
                    break
                
    plt.figure()
    plt.plot(losses)
    plt.plot(valLosses)
    plt.ylabel('Loss')
    plt.show()
    
    
if __name__ == "__main__":
    main()