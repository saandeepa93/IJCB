import torch 
from torch import nn 
from torch.nn import functional as F

from einops import rearrange, reduce, repeat

from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from icecream import ic
from sys import exit as e
import time



def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps

def bhatta_coeff(z1, z2):
  return 0.5 * (z1 + z2)

class FlowConLoss:
  def __init__(self, cfg, n_bins, device):
    self.cfg = cfg
    self.n_bins = n_bins
    self.device = device
    self.n_pixel = cfg.DATASET.IMG_SIZE * cfg.DATASET.IMG_SIZE * cfg.FLOW.N_CHAN
    self.init_loss = -log(n_bins) * self.n_pixel
    self.temp = torch.tensor(cfg.MODEL.TEMP)
    
  
  def nllLoss(self, out, logdet, means, log_sds):
    b_size, _, _, _ = out[0].size()

    # Calculate total log_p
    log_p_total = 0
    log_p_all = torch.zeros((b_size, b_size), dtype=torch.float32, device=self.device)
    log_p_nll = 0

    for i in range(self.cfg.FLOW.N_BLOCK):
      z = out[i]
      mu = means[i]
      log_sd = log_sds[i]
      
      # Create mask to select NLL loss elements
      b, c, h, w = z.size()
      z = z.view(b, 1, c, h, w)
      nll_mask = torch.eye(b, device=self.device).view(b, b, 1, 1, 1)
      nll_mask = nll_mask.repeat(1, 1, c, h, w)

      # Square matrix for contrastive loss evaluation      
      log_p_batch = gaussian_log_p(z, mu, log_sd)

      # NLL losses
      log_p_nll_block = (log_p_batch * nll_mask).sum(dim=(2, 3, 4))
      log_p_nll_block = log_p_nll_block.sum(dim=1)
      log_p_nll += log_p_nll_block

      log_p_all += log_p_batch.sum(dim=(2, 3, 4))


    logdet = logdet.mean()
    loss = self.init_loss + logdet + log_p_nll
    return ( 
      (-loss / (log(2) * self.n_pixel)).mean(),
      (log_p_nll / (log(2) * self.n_pixel)).mean(),
      (logdet / (log(2) * self.n_pixel)).mean(), 
      (log_p_all/ (log(2) * self.n_pixel))
      # log_p_nll
  )
  
  def conLoss(self, log_p_all, labels):
    b, _ = log_p_all.size()
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(self.device)

    tau = self.temp
    # Create similarity and dissimilarity masks
    off_diagonal = torch.ones((b, b), device=self.device) - torch.eye(b, device=self.device)
    sim_mask = (mask @ mask.T) * off_diagonal
    diff_mask = (1. - sim_mask ) * off_diagonal

    # Get respective log Probablities to compute row-wise pairwise against b*b log_p_all matrix
    diag_logits = (log_p_all * torch.eye(b).to(self.device)).sum(dim=-1)

    # Compute pairwise bhatta coeff. (0.5* (8, 8) + (8, 1))
    pairwise = (0.5 * (log_p_all.contiguous().view(b, b) + diag_logits.view(b, 1)))
    pairwise_exp = torch.div(torch.exp(
      pairwise - torch.max(pairwise, dim=1, keepdim=True)[0]) + 1e-5, self.temp)
    
    pos_count = sim_mask.sum(1)
    pos_count[pos_count == 0] = 1

    log_prob = pairwise_exp - (pairwise_exp.exp() * off_diagonal).sum(-1, keepdim=True).log()

    # compute mean against positive classes
    mean_log_prob_pos = (sim_mask * log_prob).sum(1) / pos_count
    return -mean_log_prob_pos



class FocalLoss(nn.Module):
  def __init__(self, args, n_bins, cfg, device):
    super().__init__()
    self.gamma = 1.
    self.n_bins = n_bins
    self.cfg = cfg
    self.device = device
    self.n_pixel = cfg.DATASET.IMG_SIZE * cfg.DATASET.IMG_SIZE * cfg.FLOW.N_CHAN
    ckp = f"server_{args.config.split('_')[0]}"
    self.means = torch.load(f"./data/{ckp}_mu_cls.pt", map_location = device)
    self.log_sds = torch.load(f"./data/{ckp}_sd_cls.pt", map_location = device)
    
  
  def forward(self, input, target, zs):
    # Calculate cross entropy and log likelihood for each class
    ce_loss = F.cross_entropy(input, target, reduction='none')
    log_likelihood = self.calc_likelihood(zs)
    loss = F.nll_loss(F.softmax(log_likelihood, dim=-1), target)
    return loss, log_likelihood
    # Generate weight matrix for each training example
    target_vec = F.one_hot(target, num_classes = self.cfg.DATASET.N_CLASS)
    ll = F.softmax(log_likelihood, dim=1)
    weights = (target_vec * ll).sum(dim=1)

    # Calculate final loss; variant of focal loss
    final_loss = torch.pow((1 - weights), self.gamma) * ce_loss

    return final_loss.mean(), log_likelihood

  def calc_likelihood(self, out):
    b_size, _, _, _ = out[0].size()
    log_p_all = torch.zeros((b_size, self.cfg.DATASET.N_CLASS), dtype=torch.float32, device=self.device)
    for i in range(self.cfg.FLOW.N_BLOCK):
      z = out[i]
      z = rearrange(z, 'b c h w -> b 1 c h w')
      mu = self.means[i]
      log_sd = self.log_sds[i]
      log_p_batch = gaussian_log_p(z, mu, log_sd)
      log_p_all += log_p_batch.sum(dim=(2, 3, 4))

    return log_p_all/ (log(2) * self.n_pixel)


