from operator import itemgetter
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import silhouette_score,adjusted_rand_score,homogeneity_score,normalized_mutual_info_score,adjusted_mutual_info_score,calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import f1_score,accuracy_score
import umap
import warnings
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from operator import itemgetter

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='sim1', help='used dataset name')
parser.add_argument("--supervised", type=bool, default=0, help='True for stage 2 validation and False for stage1')
args = parser.parse_args()
warnings.filterwarnings("ignore")


if args.dataset == 'sim1':
    label_dict = {'6_1':0,'6_2':1,'6_3':2,'6_4':3,'6_5':4}
elif args.dataset == 'sim2':
    label_dict = {'9_1':0,'9_2':1,'9_3':2,'9_4':3,'9_5':4,'9_6':5,'9_7':6,'9_8':7}
elif args.dataset == 'sim3':
    label_dict = {'13_1':0,'13_2':1,'13_3':2,'13_4':3,'13_5':4,'13_6':5,'13_7':6,'13_8':7,'13_9':8,'13_10':9,'13_11':10,'13_12':11}


def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 


label_dict_ = {}
for key, value in label_dict.items():
    label_dict_[value] = key
    



if not args.supervised:

    print('cell clustering validation')
    Y_label = np.load('./datasets/example_dataset/lbls.npy')
    x_ach = np.load('./output/scMHNN_embedding.npy')
    # print(Y_label.shape)
    # print(x_ach.shape)

    ASW_learned = []
    PS_learned = []
    HS_learned = []

    embedding_name = []
    cell_name = []
    for i in range(x_ach.shape[1]):
        embedding_name.append(str(i))

    for i in range(x_ach.shape[0]):
        cell_name.append(str(i))
    embedding_name = pd.DataFrame(index=embedding_name)
    cell_name = pd.DataFrame(index=cell_name)
    adata_learned=ad.AnnData(x_ach,obs=cell_name,var=embedding_name)
    Y_label_list = itemgetter(*list(Y_label))(label_dict_)
    adata_learned.obs['cell_type'] = Y_label_list
    sc.pp.neighbors(adata_learned,use_rep='X')
    sc.tl.umap(adata_learned,n_components=2)
    sc.pl.umap(adata_learned, color="cell_type")
    plt.title('scMHNN')
    plt.savefig('./output/scMHNN_umap_{}.jpg'.format(args.dataset), bbox_inches='tight', dpi=800)

    for i in list(np.round(np.linspace(0.1,1.0,10),1)):
        print('resolution:',i)

        sc.tl.louvain(adata_learned,resolution = i,key_added = "louvain")  # best
        y_predict = adata_learned.obs['louvain']

        ASW_learned.append(np.round(silhouette_score(x_ach,y_predict),3))
        PS_learned.append(np.round(purity_score(Y_label,y_predict),3))
        HS_learned.append(np.round(homogeneity_score(Y_label,y_predict),3))

    print('cell clustering results:')
    print('ASW_learned = ',ASW_learned)
    print('PS_learned = ',PS_learned)
    print('HS_learned = ',HS_learned)

else:
    print('cell annotation validation')
    df_result = pd.DataFrame()
    lbls_test = np.load('./output/lbls_test.npy') 
    preds_test = np.load('./output/preds_test.npy')

    df_result['acc'] = [np.round(accuracy_score(lbls_test,preds_test),3)]
    df_result['f1w'] = [np.round(f1_score(lbls_test,preds_test, average='weighted'),3)]
    print(df_result)


