import torch
from torch import nn
from .layer import HGNN_conv
import torch.nn.functional as F
import random
from scipy.sparse import coo_matrix
import numpy as np


# scMHNN for stage1
class HGNN_unsupervised(nn.Module):
    def __init__(self, in_ch, n_hid, dropout):
        super(HGNN_unsupervised, self).__init__()
        self.dropout = dropout

        # define HGNN encoder
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

        # define mlp encoder
        self.mlp1 = nn.Linear(in_ch, n_hid)
        self.mlp2 = nn.Linear(n_hid, n_hid)


    def forward(self, x, G):
        
        x_pos = F.relu(self.mlp1(x))
        x_pos = F.dropout(x_pos, self.dropout)
        x_pos = self.mlp2(x_pos)

        x_ach = F.relu(self.hgc1(x, G))
        x_ach = F.dropout(x_ach, self.dropout)
        x_ach = self.hgc2(x_ach, G)

        x_neg = x_pos[torch.randperm(x_pos.size()[0])]
        return x_ach, x_pos, x_neg


# scMHNN for stage2
class HGNN_supervised(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN_supervised, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc3(x, G)
        return x


def generate_node_pair_sets(H_rna, H_adt, H_atac):
        

    np.fill_diagonal(H_rna, 0)
    np.fill_diagonal(H_adt, 0)
    np.fill_diagonal(H_atac, 0)


    H_rna = np.where(H_rna,1,0)
    H_adt = np.where(H_adt,1,0)
    H_atac = np.where(H_atac,1,0)
    H_all = H_rna + H_adt + H_atac
    H_tri = np.where(H_all==3,1,0)
    H_bi = np.where(H_all==2,1,0)
    H_single = np.where(H_all==1,1,0)
    H_none = np.where(H_all==0,1,0)



    return H_tri, H_bi, H_single, H_none




def neighbor_sampling(H,positive_neighbor_num,p):
    # Given a dense incidence matrix and a sample num (positive_neighbor_num*p)
    # Return a sampled coordinate array
    row_coor, col_coor = np.nonzero(H)
    coor = np.vstack((row_coor,col_coor))
    indices = list(range(coor.shape[1]))
    random.shuffle(indices)
    num_subset = int(positive_neighbor_num*p)
    idx_subset = indices[:num_subset]
    coor_sampled = coor[:,idx_subset]

    return coor_sampled


def neighbor_concat(coor_sampled_tri,coor_sampled_bi,coor_sampled_single,N):
    # Given three sets of sample neighbors from three types
    # Return a dense indicator matrix
    
    coor = np.hstack((coor_sampled_tri,coor_sampled_bi,coor_sampled_single))
    data = np.ones(coor.shape[1])
    return coo_matrix((data,(coor[0,:],coor[1,:])),shape=(N,N)).toarray()
