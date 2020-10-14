import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
from sklearn.isotonic import IsotonicRegression
from time import perf_counter 
from tqdm import tqdm as pbar


def _delta(means,stds,ys):
 
    N = means.shape[0]

    #calculates the extra part that that should be subtracted
    dist = Normal(means,stds)
    cdf  = dist.cdf(ys)
    pdf  = dist.log_prob(ys).exp()
    sorted_cdf,ind = cdf.sort()
    sorted_pdf = pdf[ind]
    sorted_stds = stds[ind]
    num   = sorted_stds * sorted_pdf
    denom_diff = N*(sorted_cdf[1:] - sorted_cdf[:-1] + 1e-4)
    num_diff   = num[1:] - num[:-1]

    ans1       = num_diff/denom_diff
    delta1     = ans1.sum()
    delta2     = (sorted_stds[0] * sorted_pdf[0])/( N * sorted_cdf[0])
    delta      = (delta1 + delta2)

    return delta                                   
                                       








def _fit_isotonic(model,train_loader):

    t_start = perf_counter()
    means,stds,ys = model.mc_prediction_loader(train_loader)
    N = means.shape[0]    

    dist  = Normal(means,stds)
    cdf   = dist.cdf(ys)
    sorted_cdf,ind = cdf.sort()  #[N]
    y  = torch.arange(1.0,N+1)/N #[N]
    

    ir = IsotonicRegression(out_of_bounds='clip')
    x  =  sorted_cdf.cpu().numpy() #[N]
    y  =  y.numpy() #[N]

    x_app = np.insert(x,0,0.0)
    y_app = np.insert(y,0,0.0)
    y_ = ir.fit_transform(x_app, y_app)#[N]
    delta = _delta(means,stds,ys)

    #for synchronizing cuda calls
    torch.cuda.synchronize()
    #stop and measure the time taken for postprocessing method
    t_stop = perf_counter()
    iso_time = torch.tensor(t_stop - t_start)
    return ir,delta,sorted_cdf,iso_time
