import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression



 
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
