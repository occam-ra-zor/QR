import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression



def Neural_Sort(scores , tau=0.001):
    """
          scores : torch Tensor 
                shape : [N,1]
                
          returns : torch .Tensor
                  shape : 
    """
    scores = scores.unsqueeze(dim = 0) #[1,N,1]
    bsize = scores.size()[0]
    dim = scores.size()[1]
    one = torch.cuda.DoubleTensor(dim, 1).fill_(1)

    A_scores = torch.abs(scores - scores.permute(0, 2, 1))
    B = torch.matmul(A_scores, torch.matmul(
        one, torch.transpose(one, 0, 1)))
    scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
               ).type(torch.cuda.DoubleTensor)
    C = torch.matmul(scores, scaling.unsqueeze(0))

    P_max = (C-B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return (P_hat @ scores).squeeze(dim=0).squeeze(dim=-1)


def  ckl_divergence(c):
    """
       c : torch Tensor
       shape : [N,1]
    """
    
    c += 1e-8
    c  = Neural_Sort(c)
    N  = c.shape[0]
    
    a1 = (torch.arange(1.0,N).to(device))/N
    a2 = torch.log(a1)
    a3 = c[:-1]- c[1:]
    a  = (a1 * a2 * a3).sum()
    
    b1 = (1-c)
    b2 = torch.log(b1)
    b  = (b1 * b2).mean()
    loss = a+b + 0.5
    
    return loss

 
def l2_calibration_loss(pred_cdf,n_bins=10):
    
    with torch.no_grad():
        N    = pred_cdf.shape[0]
        bin_quantiles      = torch.zeros(n_bins)
        bin_boundaries     = torch.linspace(0,1,n_bins+1)[1:]
        for j,y in enumerate(bin_boundaries):
            bin_quantiles[j] = (pred_cdf[pred_cdf <= y]).shape[0]

        bin_quantiles      =   bin_quantiles/N
        bin_diff           =   (bin_quantiles - bin_boundaries)**2
        calibration_loss   =   bin_diff.sum() 
        
    return calibration_loss