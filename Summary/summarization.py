#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:07:11 2020

@author: cory
"""
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from torchtext import data
from torch.nn import functional as F
from torch.optim import Adam, RMSprop
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random 
import spacy
import os
import tqdm
from lstm import Encoder, Decoder 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def word_ids_to_sentence(id_tensor, vocab, join=None):
    batch = [vocab.itos[ind] for ind in id_tensor] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)
    
    

def main():
    epochs = 3
    batchSize = 8
    lr = 0.0001
    
    
    #writer = SummaryWriter('./logs')
    #train = pd.read_csv(f'train_sam.csv')
    #train.columns = ["article", "title"]
    TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), lower=True, init_token = '<sos>', eos_token='<eos>')
    trn_data_fields = [("original", TEXT),
                   ("summary", TEXT)]
    train, valid = data.TabularDataset.splits(path=f'',
                                     train='train_sam.csv', validation='valid_sam.csv',
                                     format='csv', skip_header=True, fields=trn_data_fields)
    
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    
    train_iter, val_iter = data.BucketIterator.splits((train, valid),batch_sizes=(batchSize, batchSize),sort_key=lambda x: len(x.original), sort_within_batch=False, repeat=False, device = device)

    input_size = len(TEXT.vocab)
    hidden_size = 256
    dropout = 0.5
    num_layers = 2
    
    
    
    
    bidirectional = True
    
    encoder = Encoder(input_size, hidden_size, num_layers, batchSize,bidirectional,device).to(device)
    decoder = Decoder(input_size, hidden_size, num_layers, dropout, batchSize,device).to(device)
    
    # define your LSTM loss function here
    #loss_func = F.cross_entropy()

    # define optimizer for lstm model
    #optim = Adam(model.parameters(), lr=lr)
    encoder_optimizer = Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = Adam(decoder.parameters(), lr=lr)
    losses = []
    valLosses = []
    originals = []    
    
    criterion = nn.NLLLoss()
    
    for epoch in range(epochs):
        originals = []
        genSumms = []
        origSumms= []
        batchNum = 0
        step = 0
        valBatchNum = 0
        totLoss = 0
        index  = 0 
        for batch in train_iter:
            #print("test")
            batchS = len(batch)
            batchNum+=1
            loss = 0
            
            orig = batch.original
            summ = batch.summary
            
            encoder_outputs = torch.zeros(100, encoder.hidden_size *(1+int(bidirectional)),device=device)
            encoder_hidden = encoder.initHidden(batchS)           
            encoder_outputs, encoder_hidden = encoder.forward(orig, encoder_hidden)
            
            decoder_hidden = encoder_hidden
            decoder_input = torch.ones(batchS, device=device).long()*2
            
            decoder_context = torch.zeros(batchSize, decoder.hidden_size*2).to(device)
           #print(decoder_input.data)
            genSumm = []
            origSumm = []
            genSumm.append(2)
            
            
            #print(decoder_hidden[0].shape)
            teacher_forcing = True if random.random()<0.5 else False
            if teacher_forcing:
                for di in range(len(summ)-1):
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder.forward(decoder_hidden, decoder_context,
                            encoder_outputs, decoder_input)
                    #print(decoder_output[0].data)
                    DO = decoder_output.detach().cpu().numpy()
                    value = np.argmax(DO[0])
                    genSumm.append(value)
                    
                    loss += criterion(decoder_output, summ[di+1])
                    
                    decoder_input = summ[di+1]  
                    
                    origSumm.append(summ[di][0])
                    #print(np.argmax(DO[5]))
                
            else:
                for di in range(len(summ)-1):
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder.forward(decoder_hidden, decoder_context,
                            encoder_outputs, decoder_input)
                    topv, topi = decoder_output.topk(1)
                    newIn = topi[0][0]
                    decoder_input = topi.squeeze().detach()
                    loss += criterion(decoder_output, summ[di+1])

                    DO = decoder_output.detach().cpu().numpy()
                    genSumm.append(np.argmax(DO[0]))

                    origSumm.append(summ[di][0])
                    
                
            lossAvg = loss.item()/len(summ)
                
            totLoss+=lossAvg

            encoder_optimizer.zero_grad()
            
            decoder_optimizer.zero_grad()
            
            loss.backward()
            
            #writer.add_scalar('training loss', loss.item(), step+1)
            step +=1
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),5)
            
            encoder_optimizer.step()
            decoder_optimizer.step()
            genSumms.append(genSumm)
            origSumms.append(origSumm)
            
            originals.append(orig[:,0])
            genTensorO = torch.IntTensor(origSumms[index])
            genTensor = torch.IntTensor(genSumms[index])

            if (batchNum % 50 == 0):
                avgLoss = totLoss/50
                losses.append(lossAvg)
                print("Epoch: [{}/{}], Batch:[{}/{}], Loss: {}".format(
                        epoch, epochs, batchNum, len(train_iter),avgLoss))
                totLoss= 0
                with open("/content/drive/My Drive/Summary/out.txt","a") as myfile:
                    myfile.write(str(avgLoss)+"\n")

            if(batchNum % 100 == 0):
                translatedOrig = word_ids_to_sentence(originals[index],TEXT.vocab,join = ' ')
                print(translatedOrig)
                translatedSummO = word_ids_to_sentence(genTensorO,TEXT.vocab,join = ' ')
                print(translatedSummO)
                translatedSumm = word_ids_to_sentence(genTensor,TEXT.vocab,join = ' ')
                print(translatedSumm)
                with open("/content/drive/My Drive/Summary/summout.txt","a") as myfile:
                    myfile.write(translatedOrig+" , ")
                    myfile.write(translatedSummO+ " , ")
                    myfile.write(translatedSumm+"\n")

                index+=100
               #genSumms = []
#           
            if (batchNum % 50 == 0):
                for batchVal in val_iter:
                    valBatchNum+=1
                    valLoss = 0
                    batchS = len(batchVal)
                    #valOrig = batchVal.original
                    valSumm = batchVal.summary
                    orig = batchVal.original
                    encoder_outputs = torch.zeros(100, encoder.hidden_size *(1+int(bidirectional)),device=device)
                    encoder_hidden = encoder.initHidden(batchS)           
                    encoder_outputs, encoder_hidden = encoder.forward(orig, encoder_hidden)
                    
                    decoder_hidden = encoder_hidden
                    decoder_input = torch.ones(batchS, device=device).long()*2
                    
                    decoder_context = torch.zeros(batchSize, decoder.hidden_size*2).to(device)
                  #print(decoder_input.data)
                    #genSumm = []
                    #valSumm = []
                    #genSumm.append(2)
                    
                    teacher_forcing = True if random.random()<0.5 else False
                    if teacher_forcing:
                        for di in range(len(valSumm)-1):
                            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder.forward(decoder_hidden, decoder_context,
                                    encoder_outputs, decoder_input)
                            #print(decoder_output[0].data)
                            DO = decoder_output.detach().cpu().numpy()
                            value = np.argmax(DO[0])
                            genSumm.append(value)
                            
                            valLoss += criterion(decoder_output, valSumm[di+1])
                            
                            decoder_input = valSumm[di+1]  
                            
                            origSumm.append(valSumm[di][0])
                            #print(np.argmax(DO[5]))
                        
                    else:
                        for di in range(len(valSumm)-1):
                            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder.forward(decoder_hidden, decoder_context,
                                    encoder_outputs, decoder_input)
                            topv, topi = decoder_output.topk(1)
                            newIn = topi[0][0]
                            decoder_input = topi.squeeze().detach()
                            valLoss += criterion(decoder_output, valSumm[di+1])

                            DO = decoder_output.detach().cpu().numpy()
                            genSumm.append(np.argmax(DO[0]))

                            origSumm.append(valSumm[di][0])
                    valLossAvg = valLoss.item()/len(valSumm)
                
                    valLosses.append(valLossAvg)
                    print("VALLoss: {}".format(valLossAvg))
                    with open("/content/drive/My Drive/Summary/valout.txt","a") as myfile:
                        myfile.write(str(valLossAvg)+"\n")
                    break
                    
        model1_name = "encoder.pt"
        path = F"/content/drive/My Drive/Summary/{model1_name}"
        torch.save(encoder.state_dict(),path)    
        
        model2_name = "decodeer.pt"
        path = F"/content/drive/My Drive/Summary/{model2_name}"
        torch.save(decoder.state_dict(),path) 

    plt.figure()
    plt.plot(losses)
    plt.plot(valLosses)
    plt.ylabel('Loss')
    plt.show()
    
    
if __name__ == "__main__":
    main()