import torch
import torch.nn as nn
import numpy as np
from utils.calibration_utils import l2_calibration_loss
from torch.distributions.normal import Normal
from bisect import bisect_left 



def save_stats(list_of_stats,name,qr = False,K=0.0):
    """
      stats : list of stats
    """
    arr  = np.array(list_of_stats) #[N,8]
    mean = arr.mean(axis = 0) #[N]
    std  = arr.std(axis=0) #[N]
    
    #indices for corresponding stats 
    
    #print(list_of_stats)
    rmse      = 0
    iso_rmse  = 1
    nll       = 2
    iso_nll   = 3
    calib     = 4
    iso_calib = 5
    time      = 6
    iso_time  = 7
    
    
    print("calib         : {0:0.2f} -+ {1:0.2f}".format(mean[calib]      , std[calib]     ))
    print("iso_calib     : {0:0.2f} -+ {1:0.2f}".format(mean[iso_calib]  , std[iso_calib] ))
    print("rmse          : {0:0.2f} -+ {1:0.2f}".format(mean[rmse]       , std[rmse]      ))
    print("iso_rmse      : {0:0.2f} -+ {1:0.2f}".format(mean[iso_rmse]   , std[iso_rmse]  ))
    print("nll           : {0:0.2f} -+ {1:0.2f}".format(mean[nll]        , std[nll]       ))
    print("iso_nll       : {0:0.2f} -+ {1:0.2f}".format(mean[iso_nll]    , std[iso_nll]   ))
    print("time          : {0:0.2f} -+ {1:0.2f}".format(mean[time]       , std[time]      ))
    print("iso_time      : {0:0.2f} -+ {1:0.2f}".format(mean[iso_time]   , std[iso_time]  ))
    
    N = mean.shape[0]
    data = np.concatenate([mean,std])
    #np.savetxt("experimental_results/temp"+str(name)+"_K="+str(K)+".txt", data, newline=" ")
    #np.savetxt("experimental_results/"+str(name)+"_K="+str(K)+".txt", std , newline=" ")
    #print("Done.stats are saved to {}".format("experimental_results/"+str(name)+"_K="+str(K)+".txt"))

def display_stats(stats):
    """
       stats : [rmse_before,rmse_after,nll_before,nll_after,calib_before,calib_after,training time,post-hoc time]
       and print them accordingly
    """
    
    rmse_before  = stats[0]
    rmse_after   = stats[1]
    nll_before   = stats[2]
    nll_after    = stats[3]
    calib_before = stats[4]
    calib_after  = stats[5]
    time         = stats[6]
    iso_time     = stats[7]
   
    
    print("rmse                - before : {}                after : {}".format(rmse_before , rmse_after))
    print("nll                 - before : {}                after : {}".format(nll_before , nll_after))
    print("L2_calib            - before : {}                after : {}".format(calib_before,calib_after))
    print("train time          - before : {} post-processing time : {}".format(time,iso_time))
   

def avg_stats(list_of_stats):
    """
        given list of stats find the averages 
        used when we are k-fold splitting and multiple runs
        stats : [rmse_before,rmse_after,nll_before,nll_after,calib_before,calib_after,training time,post-hoc time]
    """
    arr   = np.array(list_of_stats)#[N,9]
    N    = len(list_of_stats)
    mean = arr.mean(axis = 0)#[9]
    return mean.tolist()
    

class tester():
    
    def __init__(self,network,delta,iso,t_cdf,mean,std,qr_reg = False,K=0.0,device='cuda'):
        """
            newtwork : nn-module
            delta    : int
            ir       : isotonic regression object
            icdf     : torch Tensor will be conerted to list
            m        : mean used for normalization of outputs
            s        : std  used for normalization of outputs 
            
        """
        self.network = network
        self.qr_reg = qr_reg
        self.K = K
        self.delta = delta
        self.iso   = iso
        self.t_cdf  = t_cdf
        self.device = device
        self.s = std
        self.m = mean
        self.network.eval()
        
        
        
    def rmse(self,pred,target):
        """
             pred : torch Tensor
                         shape : [N]
             target : torch Tensor
                         shape : [N]
                         
             ------------------------
             handling normalization :
             rmse = s* rmse
                 shape : [1]
        """
        
        se     = (pred-target)**2
        mse    = se.mean()
        rmse   =  torch.sqrt(mse)
        f_rmse = (self.s * rmse).unsqueeze(dim=-1)
        return f_rmse.item()
    
    def nll_before(self,mu,sigma,target):
        """
                mu : torch Tensor
                        shape : [N]
                sigma : torch Tensor
                        shape : [N]
                target : torch Tenosr
                        shape : [N]
                ------------------------
                returns nll
                
                to handle normalization 
                n * log (s)  but we have already
                taken mean of likelihood which implies that 
                we need only add  nll+log(s)
        """
        
        dist = Normal(mu,sigma)
        nll  = -1 * dist.log_prob(target).mean()
        f_nll = nll + torch.log(self.s)
        return f_nll.item()
        
    def nl_after(self,means,stds,ys):
        """
            negative log likelihood  after iso transformation can be 
            found by doing binary search
            
            means : [M] [test_size]
            stds  : [M] [test_size]
            ys    : [M] [test_size]
            
            ---------------------------------------------
            returns 
            nll : [M] test_size
            count : number of times likelihood is assigned zeor( nll is inf)
        """
        
        



        #get test size
        M = means.shape[0]
        #get train size
        N = self.t_cdf.shape[0]
        
        #convert train_cdf from torch tensor to list for binary search
        t_cdf_list = self.t_cdf.cpu().squeeze().tolist()
        # add  C_{0} = 0.0 useful for binary search procedure
        #t_cdf doesn't have C_{0} by default 
        # we add (0.0,0.0) while fitting isotonic regression and also while 
        # getting updated nll
        
        t_cdf_list = np.trim_zeros(t_cdf_list,'f')
        t_cdf_list = [0.0]+ t_cdf_list
            
          
        #first posit normal distribution
        dist = Normal(means,stds)
        #get likelihood before transformation
        log_fx   = dist.log_prob(ys)
        #get distribution function before transformation
        Fx   = dist.cdf(ys)
        #print("Fx", Fx)
        #array that stores updated likelihood
        fx_up = torch.zeros(M)
        #array that stores which are out of support
        count = 0
        
        #we can't do anything but iterative procedure  : time complexity N log M  
        for i in range(M): 
            
            #do binary search to position of point 
            #note that there is real possibility of assigning zero likelihood
            #so we should handle it carefully
            
            left = bisect_left(t_cdf_list,Fx[i].item())
            
            #finding amount by which should be subtracted based on the position
            
            if (t_cdf_list[-1] < Fx[i]) :
                # if F(x) is greater  than last point in calibration dataset
                #then increase the 'out_of_support' points
                count = count+1
                fx_up[i] = 0.0
                
            else :   
            
                if(left == 0):
                    log_sub = torch.log(N* (self.t_cdf[1] - self.t_cdf[0] + 1e-8))
                elif left < N:
                    log_sub = torch.log( N*(self.t_cdf[left] - self.t_cdf[left-1] + 1e-8) )
                else :
                    log_sub = torch.log( N* (self.t_cdf[left-1] - self.t_cdf[left-2] + 1e-8) )
                    
                fx_up[i] = (log_fx[i] - log_sub).exp()

               
            
 
        f_nl     = -1*(fx_up.mean() / self.s)
        return f_nl.item(),fx_up,count
                
    
    def calib_error_after(self,cdf):
        """
             after iso transformation 
             for computing calibration error cdf values should be 
             composed with isotonic regression and then must be computed
             
             ----------------------------
             handling normalization
             l_2 error doesn't change with location-scale transformation
             
             cdf : [N]
        
        """
        
        cdf = cdf.cpu().detach().numpy() 
        transformed_cdf = torch.from_numpy(self.iso.transform(cdf)).to(self.device)
        calib_error  = l2_calibration_loss(transformed_cdf)
        return calib_error.item()

    
    def test(self,test_loader):
        
        """
                main function of tester object 
                testloader to get test_data
                
                --------------------------------------
                returns :
                stats list[rmse_before,rmse_after,nll_before,nll_after,calib_before,calib_after,count]
                count 
        
        """
        
        with torch.no_grad():
            
            means,stds,ys = self.network.mc_prediction_loader(test_loader)# shapes are [test_size],[test_size],[test_size]
            means_after   = means - (stds* self.delta) #[test_size]
            
            
            d = torch.nonzero(torch.isnan(means_after.view(-1)))
            test_dist     = Normal(means,stds)
            test_cdf      = test_dist.cdf(ys)

            rmse_before = self.rmse(means,ys)
            rmse_after  = self.rmse(means_after,ys)

            nll_before              = self.nll_before(means,stds,ys)
            nl_after,fx_up,count         = self.nl_after(means,stds,ys)
            
            calib_before = 100*l2_calibration_loss(test_cdf)
            calib_after  = 100*self.calib_error_after(test_cdf)

            stats = [rmse_before,rmse_after,nll_before,nl_after,calib_before,calib_after]           
            return stats,fx_up,count
    
               