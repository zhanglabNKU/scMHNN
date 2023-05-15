from utils import *
import numpy as np



# load RNA modality data 
def load_rna_data(data_dir,lbls_dir):
    fts = np.load(data_dir)
    print('rna fts shape:',fts.shape)
    lbls = np.load(lbls_dir)
    return fts, lbls


# load Protein modality data 
def load_adt_data(data_dir):
    fts = np.load(data_dir)
    print('protein fts shape:',fts.shape)
    return fts


# load ATAC modality data 
def load_atac_data(data_dir):
    fts = np.load(data_dir)
    print('atac fts shape:',fts.shape)
    return fts


def load_feature_and_H(data_dir_rna,
                     data_dir_adt,
                     data_dir_atac,
                     lbls_dir,
                     m_prob=1,
                     K_neigs=[10],
                     is_probH=True,
                     edge_type = 'pearson',
                     use_rna = True,
                     use_adt = True,
                     use_atac = True):

    
    if use_rna:
        ft_rna, lbls = load_rna_data(data_dir_rna, lbls_dir)

    if use_adt:
        ft_adt = load_adt_data(data_dir_adt)   

    if use_atac:
        ft_atac = load_atac_data(data_dir_atac)

    fts = None
    if use_rna:
        fts = Multi_omics_feature_concat(fts, ft_rna)
    if use_adt:
        fts = Multi_omics_feature_concat(fts, ft_adt)
    if use_atac:
        fts = Multi_omics_feature_concat(fts, ft_atac)
    print('fts shape:',fts.shape)


    print('Constructing the multi-omics hypergraph incidence matrix!')
    H = None

    if use_rna:
        H_rna = construct_H_with_KNN(ft_rna, K_neigs=K_neigs,is_probH=is_probH, m_prob=m_prob, edge_type=edge_type)
        H = Multi_omics_hyperedge_concat(H, H_rna)

    
    if use_adt:
        H_adt = construct_H_with_KNN(ft_adt, K_neigs=K_neigs,is_probH=is_probH, m_prob=m_prob, edge_type=edge_type)
        H = Multi_omics_hyperedge_concat(H, H_adt)

    
    if use_atac:
        H_atac = construct_H_with_KNN(ft_atac, K_neigs=K_neigs,is_probH=is_probH, m_prob=m_prob, edge_type=edge_type)
        H = Multi_omics_hyperedge_concat(H, H_atac)

    print('Finish the Construction of hypergraph incidence matrix!')
    

    return fts, lbls, H, H_rna, H_adt, H_atac



