import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout_VI(nn.Module):
    def __init__(self, input_dim , output_dim , num_units, drop_prob):
        
        super(Dropout_VI , self).__init__()
        
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.num_units     = num_units
        self.drop_prob     = drop_prob
        
        self.layer1        = nn.Linear(input_dim, num_units)
        self.layer2        = nn.Linear(num_units, num_units)
        self.output        = nn.Linear(num_units,output_dim*2)
        
        self.activation1   = nn.ReLU()
        self.activation2   = nn.ReLU()
        
    def mc_prediction(self,x,passes = 64):
        
        #x : torch Tensor shape [N,x_dim]
        mu    = []
        sigma = []
        N,x_dim = x.shape
        with torch.no_grad():
            for i in range(passes):
                mean,std    = self.forward(x) #[N],[N]
                mu.append(mean.unsqueeze(dim = 0))
                sigma.append(std.unsqueeze(dim = 0))
            
            mu    = torch.cat(mu , dim = 0)  #[passes,N]
            sigma = torch.cat(sigma , dim = 0)#[passes,N]
        
            mean       = mu.mean(dim = 0 )         # [N]
            std        = torch.sqrt( (sigma.mean(dim=0)**2) + mu.var(dim =0) )#[N]
        
            return mean,std
    def mc_prediction_loader(self,loader):
        
        means = []
        stds  = []
        ys    = []
        
        with torch.no_grad():
            for data in loader :
                x,y = data
                
                mu,sigma =self.mc_prediction(x)
                means.append(mu.unsqueeze(dim = -1))
                stds.append(sigma.unsqueeze(dim = -1))
                ys.append(y)
                
            means = torch.cat(means,dim = 0).squeeze(dim =-1)#[test_size]
            stds  = torch.cat(stds,dim = 0).squeeze(dim = -1)#[test_size]
            ys    = torch.cat(ys,dim = 0).squeeze(dim = -1)  #[test_size]
                
            return means,stds,ys
            
        
        
    def forward(self ,x ):
        x = x.view(-1,self.input_dim)
        x = F.dropout(self.activation1(self.layer1(x)) ,p=self.drop_prob,training= True)
        x = F.dropout(self.activation2(self.layer2(x)) ,p=self.drop_prob,training= True)
        x = self.output(x)
        mu = x[:,0]   #[N]
        sigma = 0.1 + 0.9 * F.softplus( x[:,1]) #[N]
        return mu,sigma
    
    
    
    
"""
Vanilla two hidden layer NN. which serves as base model
for deep ensembles 
"""   
class Vanilla(nn.Module):
    def __init__(self, input_dim , output_dim , num_units):
        
        super(Vanilla , self).__init__()
        
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
        x = self.output(x)
        return x
    
    
class DeepEnsemble(nn.Module):
    
    def __init__(self,input_dim,output_dim,num_units,lr ,size=5):
        super(DeepEnsemble,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = num_units
        self.size = size
        ensemble   = [Vanilla() for i in range(size)]
        optimizers = [torch.optim.Adam(ensemble[i].parameters,lr) for i in range(size)]
        
    
    
        
    def test_prediction(self,x):
        pass
        
        
        
    def train(self,data,epochs):
        pass
        
        