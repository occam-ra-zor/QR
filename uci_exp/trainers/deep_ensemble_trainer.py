import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
from sklearn.isotonic import IsotonicRegression
from time import perf_counter 
from tqdm import tqdm as pbar
from utils.qr_utils import _qr_loss
from utils.isotonic_utils import _fit_isotonic


                
class Ensemble_Trainer():
    
    
    def __init__(self,model,input_dim,batch_size, lr , qr_reg, K ,mean,std,x_range,device='cuda',spacing=64):
        
        self.model        = model 
        self.input_dim    = input_dim
        self.batch_size   = batch_size
        self.qr_reg       = qr_reg
        self.K            = K
        self.m            = mean
        self.s            = std
        self.x_range      = x_range
        self.spacing      = spacing
        
        self.ensemble_opt     = [ torch.optim.Adam(self.model.ensemble[i].parameters(), lr=lr) for i in range(5)]
        
        self.ensemble_trainer = [trainer(network = self.model.ensemble[i],input_dim =input_dim,
                                 batch_size=batch_size,optimizer = self.ensemble_opt[i],
                                 device = device,mean=self.m,std=self.s,
                                 qr_reg = qr_reg,K = K,partition = i+1,x_range=x_range,spacing=spacing) for i in range(5) ]
    
    
        
    def train(self,train_loader,epochs):
        total_time = 0.0
        for i in range(self.model.size):
            time = self.ensemble_trainer[i].train(train_loader,epochs)
            total_time += time
        return total_time
    
    
                              
    def fit_isotonic(self,train_loader):
        
        
        self.model.eval()                     
        with torch.no_grad():
            ir,delta,sorted_cdf,iso_time = _fit_isotonic(self.model,train_loader)
            return ir,delta,sorted_cdf,iso_time
            
                              
                              
            
        
        
        
        
class trainer():
   
    
    def __init__(self,network,input_dim,batch_size,optimizer,device,mean,std,partition,x_range,
                 qr_reg = False,K=0.0,spacing=64):
        self.device         = device
        self.network        = network.to(device).double()
        self.batch_size     = batch_size
        self.input_dim      = input_dim
        self.optimizer      = optimizer
        self.qr_reg         = qr_reg
        self.K              = K
        self.m              = mean
        self.s              = std
        self.partition      = partition
        self.x_range        = x_range
        self.spacing        = spacing
        
    
    
    #adversarial training
    def perturb(self,x,y):
        

        x = x.detach().clone()
        y = y.detach().clone()
        x.requires_grad = True
        y.requres_grad = False
        
        mu,sigma= self.network.forward(x)
        loss = (-1 * Normal(mu,sigma).log_prob(y.squeeze(dim = -1)) ).mean().unsqueeze(dim = -1)
        loss.backward()
        data_grad = x.grad.data
        adv_x     = self.fgsm_attack(x,data_grad)
        return adv_x    
        
    
    def fgsm_attack(self ,data, data_grad , epsilon = 0.01):
        
        sign_data_grad = data_grad.sign()
        perturbed_data = data + (epsilon * self.x_range * sign_data_grad )
        return perturbed_data.detach().clone()    
            
    def train(self,train_loader,epochs,print_freq=25):
       

        self.network.train()
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
                adv_x    = self.perturb(x,y)
                
                
                mu,sigma = self.network(x)
                loss,entropy = self._loss(mu,sigma,y.squeeze(dim = -1)) #([N,N,N] -> [1],[1])
                
                adv_mu,adv_sigma =self.network(adv_x)
                adv_loss  = (-1 * Normal(mu,sigma).log_prob(y.squeeze(dim = -1)) ).mean().unsqueeze(dim = -1)
                
                mean_loss = (loss + adv_loss)/2
                mean_loss.backward()
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
            loss = -1 * log_likelihood  +(self.K * qr_loss)
        else :
            qr_loss = torch.tensor([0.0])
            loss =  -1 * log_likelihood 

        return loss,qr_loss
    

  