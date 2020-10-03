import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Base(nn.Module):
    def __init__(self, input_dim , output_dim , num_units):
        
        super(Base , self).__init__()
        
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.num_units     = num_units
        
        self.layer1        = nn.Linear(input_dim, num_units)
        self.layer2        = nn.Linear(num_units, num_units)
        self.output        = nn.Linear(num_units,output_dim*2)
        
        self.activation1   = nn.ReLU()
        self.activation2   = nn.ReLU()
        
       
        
        
    
        
    def forward(self ,x ):
        x = x.view(-1,self.input_dim)
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.output(x)#[N,2]
        
        mu = x[:,0]   #[N]
        sigma = 0.1 + 0.9 * F.softplus( x[:,1]) #[N]
        return mu,sigma
        
    

    
    

class Ensemble():
    
    def __init__(self,input_dim,output_dim,units,device='cuda',size = 5):
        
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.units = units
        
        self.ensemble         = [Base(input_dim,output_dim,units).cuda() for i in range(5)]
        self.ensemble_opt     = [torch.optim.Adam(self.ensemble[i].parameters()) for i in range(5)]
        
    
    def eval(self):
        
        for i in range(self.size):
            self.ensemble[i].eval()
            
    def train_mode(self):
        
        for i in range(self.size):
            self.ensemble[i].train()        
            
    
    def prediction(self,x):
        with torch.no_grad():
            self.eval()
            means = []
            stds  = []
            for i in range(self.size):
                mu,sigma = self.ensemble[i](x)
                means.append(mu.unsqueeze(dim = 0))
                stds.append(sigma.unsqueeze(dim = 0))
                
            mu    = torch.cat(means , dim = 0)  #[5,N]
            sigma = torch.cat(stds , dim = 0)  #[5,N]
        
            mean       = mu.mean(dim = 0 )         # [N]
            std        = torch.sqrt( (sigma.mean(dim=0)**2) + mu.var(dim =0) )#[N]
        
            return mean,std
        
    def mc_prediction_loader(self,loader):
        
        """
        --------------------
        returns means[test_size], stds [test_size],ys [test_size]
        """
        
        means = []
        stds  = []
        ys    = []
        
        with torch.no_grad():
            for data in loader :
                x,y = data
                
                mu,sigma =self.prediction(x)
                means.append(mu.unsqueeze(dim = -1))
                stds.append(sigma.unsqueeze(dim = -1))
                ys.append(y)
                
            means = torch.cat(means,dim = 0).squeeze(dim =-1)#[test_size]
            stds  = torch.cat(stds,dim = 0).squeeze(dim = -1)#[test_size]
            ys    = torch.cat(ys,dim = 0).squeeze(dim = -1)  #[test_size]
                
            return means,stds,ys    
