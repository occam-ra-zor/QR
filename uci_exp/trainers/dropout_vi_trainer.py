import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import math
import numpy as np
from time import perf_counter 
from utils.qr_utils import _qr_loss
from utils.isotonic_utils import _fit_isotonic

class Dropout_VI_trainer():
    
    """
            Dropout VI trainer
    
    """
    
    def __init__(self,network,input_dim,batch_size,optimizer,device,mean,std,qr_reg = False,K=0.0,spacing=64):
        self.device         = device
        self.network        = network.to(device).double()
        self.batch_size     = batch_size
        self.input_dim      = input_dim
        self.optimizer      = optimizer
        self.qr_reg         = qr_reg
        self.K              = K
        self.m              = mean
        self.s              = std
        self.spacing        = spacing  
        
        
    def train(self,train_loader,epochs,print_freq=25):
        """
            training part of Dropout VI models
            mu : torch Tensor 
                shape [N]
            sigma : torch Tensor
                shape [N]
        """

        self.network.train()
        
        #starting the counter
        t_start = perf_counter() 
        
        for epoch in range(epochs):
            epoch_nll_loss = 0.0
            epoch_entropy  = 0.0
            epoch_N  =0
            
            for i,data in enumerate(train_loader):
                self.optimizer.zero_grad()
                x, y = data
                N = x.shape[0]
                epoch_N +=N
                
                mu,sigma = self.network(x)
                loss,entropy = self._loss(mu,sigma,y.squeeze(dim = -1)) #([N,N,N] -> [1],[1])
                loss.backward()
                self.optimizer.step()
                epoch_nll_loss   += (loss.detach().cpu().item() +  torch.log(self.s)) * N
                epoch_entropy    += entropy.cpu().item()
            
            epoch_nll_loss = epoch_nll_loss/epoch_N
            epoch_entropy = epoch_entropy/epoch_N

         
      
        torch.cuda.synchronize()
        t_stop = perf_counter() 
        time = torch.tensor(t_stop - t_start)
        return time
    
    
                    
    def _loss(self,mu,sigma,y):
            
        dist = Normal(mu,sigma)
        log_likelihood = dist.log_prob(y).mean().unsqueeze(dim = -1)

        if self.qr_reg :
            qr_loss = _qr_loss(dist,y,self.spacing)
            loss = -1*log_likelihood  +(self.K * qr_loss)
        else :
            qr_loss = torch.tensor([0.0])
            loss =  -1 * log_likelihood 

        return loss,qr_loss
    
    
    def fit_isotonic(self , train_loader):
        
        self.network.eval()
        with torch.no_grad():
            ir,delta,sorted_cdf,iso_time = _fit_isotonic(self.network,train_loader)
            return ir,delta,sorted_cdf,iso_time
            
           
           