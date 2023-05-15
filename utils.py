# part of codes were borrowed from 'https://github.com/iMoonLab/HGNN'

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

def Pear_corr(x):
    x = x.T
    x_pd = pd.DataFrame(x)
    dist_mat = x_pd.corr()
    return dist_mat.to_numpy()


def Eu_dis(x):
    dist_mat = cdist(x,x, 'euclid')
    print('dist_mat shape:', dist_mat.shape)
    return dist_mat


# Concat the features from multiple modalities 
def Multi_omics_feature_concat(*F_list, normal_col=False):

    features = None
    for f in F_list:
        if f is not None and f != []:
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features



# Concat the modality-specific hyperedges
def Multi_omics_hyperedge_concat(*H_list):

    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H



# generate G from incidence matrix H
def generate_G_from_H(H, variable_weight=False):

    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge, here we use 1 for each hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, edge_type = 'euclid'):

    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))


    if edge_type == 'euclid':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]

            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx


            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)

                else:
                    H[node_idx, center_idx] = 1.0
        print('use euclid for H construction')  

    elif edge_type == 'pearson':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = -999.
            dis_vec = dis_mat[center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            nearest_idx = nearest_idx[::-1]  

            avg_dis = np.average(dis_vec) 
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = 1.-np.exp(-(dis_vec[node_idx]+1.0) ** 2 )
                else:
                    H[node_idx, center_idx] = 1.0
        print('use pearson for H construction')  

    return H


def construct_H_with_KNN(X, K_neigs=[10],  is_probH=True, m_prob=1,edge_type='euclid'):

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    if edge_type == 'euclid':
        dis_mat = Eu_dis(X)

    elif edge_type == 'pearson':
        dis_mat = Pear_corr(X)

    
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob,edge_type)
        H = Multi_omics_hyperedge_concat(H, H_tmp)

    return H


def sample_(label,y):
    idx = np.where(y==label)[0]
    return np.random.choice(idx)
