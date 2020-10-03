import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import math


def Neural_Sort(scores , tau=0.001):
        """
              scores : torch Tensor 
                    shape : [N]

              returns : torch .Tensor
                      shape : [N]
        """
        
        scores = scores.unsqueeze(dim = 0).unsqueeze(dim = -1) #[1,N,1]
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

            

def _neg_entropy(c,spacing=64,device='cuda'):
    """

      c : torch Tensor 
          shape [N]

      computes negative entropy using samples 
      spacing entropy estimation.

    """


    N  = c.shape[0]
    m  = spacing
    m_spacing = c[:-m] - c[m:] #[N-m]
    scaled = -1 * torch.log(m_spacing * ((N+1)/m) )
    neg_entropy = scaled.mean().unsqueeze(dim = -1 )
    return neg_entropy




def _qr_loss(dist,y_target,spacing=64):


    cdf = dist.cdf(y_target)
    sort_cdf = Neural_Sort(cdf)
    neg_entropy = _neg_entropy(sort_cdf,spacing)
    return neg_entropy