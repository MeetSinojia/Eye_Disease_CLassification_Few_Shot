import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Function to compute the Euclidean distance between feature vectors
def euclidean_dist(x, y):

  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)


class ProtoNet(nn.Module):

  def __init__(self, encoder, device="cpu"):
    super(ProtoNet, self).__init__()

    self.device = device
    self.encoder = encoder.to(self.device)

  def set_forward_loss(self, sample):
    sample_images = sample['images'].to(self.device) # retrieve the support + query images
    n_way = sample['n_way']                 # get no. of classes in sample
    n_support = sample['n_support']         # no. of support images in each class     
    n_query = sample['n_query']             # no. of query images in each class

    x_support = sample_images[:, :n_support] 
    x_query = sample_images[:, n_support:]
   
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long() # result = torch.Size([n_way, n_query, 1])
    target_inds = Variable(target_inds, requires_grad=False) 
    target_inds = target_inds.to(self.device)
 
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
    
    z = self.encoder.forward(x) # returns embedded vector
    z_dim = z.size(-1) #get the size of the flattenned vector

    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)    
    z_query = z[n_way*n_support:]
    dists = euclidean_dist(z_query, z_proto)
    
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
   
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
   
    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat
        }