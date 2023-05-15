import torch
from torch import nn
import torch.nn.functional as F


# dual contrastive loss function

# contrastive loss1: intra-cell contrastive loss
# Takes embeddings of an anchor sample, a positive sample and a negative sample
class intra_cell_loss(nn.Module):

    def __init__(self, margin):
        super(intra_cell_loss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


# contrastive loss2: inter-cell contrastive loss
# pull similar node pairs closer and push dissimilar node pairs apart
class inter_cell_loss(nn.Module):

    def __init__(self,tau):
        super(inter_cell_loss,self).__init__()
        self.tau = tau

    def sim(self, x_ach: torch.Tensor):
        x_ach = F.normalize(x_ach)   
        return torch.mm(x_ach, x_ach.t())
  
    def forward(self, x_ach: torch.Tensor, H_union,H_none):
        f = lambda x: torch.exp(x / self.tau)
        sim_mat = f(self.sim(x_ach))
        neighbor_sim_mat = torch.mul(H_union,sim_mat)
        none_neighbor_sim_mat = torch.mul(H_none,sim_mat)
        loss = -torch.log(neighbor_sim_mat.sum()/(none_neighbor_sim_mat.sum()))
        return loss




