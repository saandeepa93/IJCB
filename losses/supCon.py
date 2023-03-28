import torch
import torch.nn as nn

from icecream import ic
from sys import exit as e


def isnan(z):
  n = torch.isnan(z).type(torch.int8).sum()
  if n > 0:
    return "YES NAN"
  else:
    return "NO NAN"

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
      super(SupConLoss, self).__init__()
      self.temperature = temperature
      self.contrast_mode = contrast_mode
      self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, contrast_count=2):
      """Compute loss for model. If both `labels` and `mask` are None,
      it degenerates to SimCLR unsupervised loss:
      https://arxiv.org/pdf/2002.05709.pdf
      Args:
          features: hidden vector of shape [bsz, n_views, ...].
          labels: ground truth of shape [bsz].
          mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
              has the same class as sample i. Can be asymmetric.
      Returns:
          A loss scalar.
      """
      device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

      batch_size = labels.shape[0]

      labels = labels.contiguous().view(-1, 1)
      mask = torch.eq(labels, labels.T).float().to(device)

      # compute logits
      anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T), 
        self.temperature
      )
      # for numerical stability
      logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
      logits = anchor_dot_contrast - logits_max.detach()

      # tile mask
      mask = mask.repeat(contrast_count, contrast_count)

      # mask-out self-contrast cases
      logits_mask = torch.scatter(
          torch.ones_like(mask),
          1,
          torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
          0
      )
      mask = mask * logits_mask

      # compute log_prob
      exp_logits = torch.exp(logits) * logits_mask
      log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

      # compute mean of log-likelihood over positive
      mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

      # loss
      loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
      
      # loss = loss.mean()
      loss = loss.view(contrast_count, batch_size).mean()

      return loss