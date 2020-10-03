import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_boston
import pandas as pd


def uci_helper(name):
    
    """
        helper function that returns the numpy
        array based on the name of dataset
    
    """
    
    if name == "airfoil":
        #[]
       
        df = pd.read_table('data/Airfoil Self Noise.dat', delim_whitespace=True,header=None)
        data = df.values
        return data
    elif name == "boston":
        #[506,14]
        X, y = load_boston(return_X_y=True)
        y = np.expand_dims(y,axis = -1)
        data = np.hstack((X,y))
        return data
        
      
    elif name == "concrete":
        df = pd.read_excel("data/Concrete Strength.xls")
        data = df.values
        return data

    
    elif name == "fish":
        df = pd.read_csv('data/Fish Toxicity.csv',header= None,delimiter=';')
        data = df.values
        return data
        
    elif name == "kin8nm":
        #[]
        df = pd.read_csv('data/Kin8nm.csv',header='infer',delimiter=',')
        data = df.values
        return data
    elif name == "protein":
        #[45730,9]
        df = pd.read_csv('data/Protein Teritary.csv',header='infer',delimiter=',')
        l  = (list(df.columns.values)[1:])
        l.append('RMSD')
        dfc = df.reindex(columns=l)
        data = dfc.values
        return data
    
    elif name == "red":
        #[1599,12]
        df = pd.read_csv('data/Wine Quality Red.csv',header='infer',delimiter=',')
        data = df.values
        return data
    
    elif name =="white":
        #[]
        df = pd.read_csv('data/Wine Quality White.csv',header='infer',delimiter=';')
        data = df.values
        return data

        
    elif name=="yacht":
        #[308,7]
        df = pd.read_table('data/Yacht Hydrodynamics.data', delim_whitespace=True, names=('CB', 'PC,', ' LDR',
                                                                'BDR' , 'LBR' , 'FN' , 'RR'))
        data = df.values
        return data
        
     
    elif name=="year":    
        #[515345,91]
        df = pd.read_csv("data/YearPredictionMSD.txt" ,header=None)
        l  = (list(df.columns.values)[1:])
        l.append(0)

        dfc = df.reindex(columns=l)
        data = dfc.values
        return data

        
    

class data_helper():
    
    """
        helper function that normalizes the data and converts to tensor
        and builds the DataLoader and returns it.
        
        also useful for return 
    """
    def __init__(self,data,train_index,test_index,input_dim,batch_size =512,device='cuda',test_batch_size=2048):
        
        self.x_train =  torch.from_numpy(data[train_index, :input_dim]).to(device).double()
        self.y_train =  torch.from_numpy(data[train_index, input_dim:]).to(device).double()
        self.x_test  =  torch.from_numpy(data[test_index,  :input_dim]).to(device).double()
        self.y_test  =  torch.from_numpy(data[test_index,  input_dim:]).to(device).double()
        
        
        self.x_mean  = self.x_train.mean(dim = 0)
        self.x_std   = self.x_train.std(dim =0)
        self.y_mean  = self.y_train.mean(dim = 0) 
        self.y_std   = self.y_train.std(dim =0)
        
        x_train = (self.x_train - self.x_mean)/ self.x_std
        y_train = (self.y_train - self.y_mean)/ self.y_std
        x_test  = (self.x_test  - self.x_mean)/ self.x_std
        y_test  = (self.y_test  - self.y_mean)/ self.y_std
        
        x_max,_ = torch.max(x_train,0) #useful for deep ensembling
        x_min,_ = torch.min(x_train,0)
        self.x_range = (x_max - x_min).view(1,-1) #[1,input_dim] imp for broadcasting while adverserial training
       
        
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset  = TensorDataset(x_test,  y_test)
        self.train_loader  = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
        self.test_loader   = DataLoader(dataset = test_dataset,batch_size =test_batch_size ,shuffle = False)
    def inverse_transform_y(self,y):
        re_norm =  (y * y_std) + y_mean
        return re_norm
        
    def y_mean(self):
        return self.y_mean
        
    def y_std(self):
        return self.y_std
    
    def train_loader(self):
        return self.train_loader
    
    def test_loader(self):
        return self.test_loader
    

    
