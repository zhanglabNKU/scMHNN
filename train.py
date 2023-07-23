import copy
import os
import torch
import torch.optim as optim
from utils import generate_G_from_H, sample_
from model.loss import intra_cell_loss,inter_cell_loss
from model.scMHNN_model import HGNN_supervised, HGNN_unsupervised, neighbor_sampling,neighbor_concat,generate_node_pair_sets
from multiomics_hypergraph_construction import load_feature_and_H
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import anndata as ad
from operator import itemgetter
import scanpy as sc


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='sim1', help='dataset name')
parser.add_argument('--data_dir_rna',  default='./datasets/example_dataset_simulation1/rna.npy', help='path of RNA data')
parser.add_argument('--data_dir_adt',  default='./datasets/example_dataset_simulation1/adt.npy', help='path of Protein data')
parser.add_argument('--data_dir_atac',  default='./datasets/example_dataset_simulation1/atac.npy', help='path of ATAC data')
parser.add_argument('--lbls_dir',  default='./datasets/example_dataset_simulation1/lbls.npy', help='path of cell lbls')
parser.add_argument("--supervised", type=bool, default=0, help='True for stage 2 (cell type annotation), False for stage1 (unsupervised cell representation learning)')
parser.add_argument("--m_prob", type=float, default=1.0, help='m_prob')
parser.add_argument("--K_neigs", type=int, default=70, help='K_neigs')  
parser.add_argument("--p_tri", type=float, default=0.8, help='sample probability for tri-neighbor set')
parser.add_argument("--p_bi", type=float, default=0.15, help='sample probability for bi-neighbor set')
parser.add_argument("--positive_neighbor_num", type=int, default=1000, help='num of node pairs in positive neighbors set')
parser.add_argument("--print_freq", type=int, default=5, help='print_freq')
parser.add_argument("--edge_type", type=str, default='euclid', help='euclid or pearson')
parser.add_argument("--is_probH", type=bool, default=False, help='prob edge True or False')
parser.add_argument("--use_rna", type=bool, default=1, help='use rna modality')
parser.add_argument("--use_adt", type=bool, default=1, help='use adt')
parser.add_argument("--use_atac", type=bool, default=1, help='use atac')
parser.add_argument("--n_hid", type=int, default=128, help='dimension of hidden layer')   
parser.add_argument("--drop_out", type=float, default=0.1, help='dropout rate')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--milestones", type=int, default=[100], help='milestones')
parser.add_argument("--gamma", type=float, default=0.9, help='gamma')
parser.add_argument("--weight_decay", type=float, default=0.0005, help='weight_decay')
parser.add_argument("--max_epoch", type=int, default=200, help='max_epoch')   
parser.add_argument("--tau", type=float, default=0.5, help='temperature Coefficient')
parser.add_argument("--alpha", type=float, default=0.05, help='balanced factor for dual contrastive loss')
parser.add_argument("--beta", type=float, default=100., help='non-negative control parameter for intra_cell_loss')
parser.add_argument("--labeled_cell_ratio", type=float, default=0.02, help='labeled cell ratio for cell type annotation')


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


fts,lbls,H,H_rna,H_adt, H_atac = load_feature_and_H(args.data_dir_rna,
                                                    args.data_dir_adt,
                                                    args.data_dir_atac,
                                                    args.lbls_dir,
                                                    m_prob=args.m_prob,
                                                    K_neigs=args.K_neigs,
                                                    is_probH=args.is_probH,
                                                    edge_type=args.edge_type,
                                                    use_rna = args.use_rna,
                                                    use_adt = args.use_adt,
                                                    use_atac = args.use_atac)



G = generate_G_from_H(H)
N = fts.shape[0]  # cell nums
n_class = int(lbls.max()) + 1   # class nums
H_tri, H_bi, H_single, H_none = generate_node_pair_sets(H_rna, H_adt, H_atac)
must_include_idx = []
for i in range(n_class):
    must_include_idx.append(sample_(i,lbls))

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
H_none = torch.from_numpy(H_none)
H_none = H_none.to(device)


def train_model_unsupervised(model, criterion_intra_cell,criterion_inter_cell, optimizer, scheduler, num_epochs=25, print_freq=500):

    loss_train = []
    epoch_all = []

    for epoch in range(num_epochs):

        epoch_all.append(epoch)
        scheduler.step()
        model.train()  
        
        coor_sampled_tri = neighbor_sampling(H_tri,args.positive_neighbor_num,args.p_tri)
        coor_sampled_bi = neighbor_sampling(H_bi,args.positive_neighbor_num,args.p_bi)
        coor_sampled_single = neighbor_sampling(H_single,args.positive_neighbor_num,1-args.p_tri-args.p_bi)
        H_union = neighbor_concat(coor_sampled_tri,coor_sampled_bi,coor_sampled_single,N)
        H_union = torch.from_numpy(H_union)
        H_union = H_union.to(device)
 
        x_ach, x_pos, x_neg = model(fts, G)
        loss_intra_cell = criterion_intra_cell(x_ach, x_pos, x_neg)
        loss_inter_cell = criterion_inter_cell(x_ach,H_union,H_none)

        loss = loss_intra_cell + args.alpha*loss_inter_cell 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss)

        torch.save(model, './output/pretrained_scMHNN.pt')

        print(f'Epoch {epoch}/{num_epochs - 1}','loss: {}'.format(loss),' loss_intra_cell: {}'.format(loss_intra_cell),' loss_inter_cell: {}'.format(loss_inter_cell))


    model.eval()
    embedding, _, _ = model(fts, G)
    return model, loss_train, epoch_all, embedding





def train_model_supervised(model, criterion, optimizer, scheduler, num_epochs, print_freq=500):


    indices_subset = list(range(N))
    random.shuffle(indices_subset)


    idx_train = indices_subset[:int(N*args.labeled_cell_ratio-n_class)]
    idx_test = indices_subset[int(N*args.labeled_cell_ratio-n_class):]
    
    new_idx_train = list(set(idx_train).union(set(must_include_idx)))
    new_idx_test = list(set(idx_test).difference(set(must_include_idx)))
    idx_train = torch.Tensor(new_idx_train).long().to(device)
    idx_test = torch.Tensor(new_idx_test).long().to(device)

    print(idx_train.shape, idx_test.shape)

    # load pretrained hypergraph encoder in stage 1
    if os.path.exists('./output/pretrained_scMHNN.pt'):
        print('loading pretrained model...')
        cpt = torch.load('./output/pretrained_scMHNN.pt')
        pretrained_dict =cpt.state_dict() 
        model_dict = model.state_dict()

        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)  

    else:
        raise FileNotFoundError("should first pretrained model in unsupervised manner with args.supervised 0")



    for epoch in range(num_epochs):

        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        scheduler.step()
        model.train()  

        running_loss = 0.0
        running_corrects = 0
        idx = idx_train 
        optimizer.zero_grad()
        outputs = model(fts, G)
        loss = criterion(outputs[idx], lbls[idx])
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * fts.size(0)
        running_corrects += torch.sum(preds[idx] == lbls.data[idx])

        epoch_loss = running_loss / len(idx)
        epoch_acc = running_corrects.double() / len(idx)

        if epoch % print_freq == 0:
            print(f' Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    model.eval()
    outputs = model(fts, G)
    _, preds = torch.max(outputs, 1)
    lbls_test = lbls[idx_test].detach().cpu().numpy()
    preds_test = preds[idx_test].detach().cpu().numpy()
    np.save('./output/lbls_test.npy', lbls_test)
    np.save('./output/preds_test.npy', preds_test)
    
    return model






if __name__ == '__main__':

    if not args.supervised:

        print('stage1: unsupervised cell representation learning')
        model_ft = HGNN_unsupervised(in_ch=fts.shape[1],n_hid=args.n_hid,dropout=args.drop_out)
        model_ft = model_ft.to(device)
        optimizer = optim.Adam(model_ft.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        schedular = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestones,gamma=args.gamma)

        criterion_intra_cell = intra_cell_loss(args.beta)
        criterion_inter_cell = inter_cell_loss(args.tau)

        model, loss_train, epoch_all, embedding = train_model_unsupervised(model_ft, criterion_intra_cell,criterion_inter_cell, optimizer, schedular, args.max_epoch, print_freq=args.print_freq)
        np.save('./output/scMHNN_embedding.npy', embedding.cpu().detach().numpy())


    else:

        print('stage1: supervised training for cell type annotation')
        model_ft = HGNN_supervised(in_ch=fts.shape[1],n_class=n_class,n_hid=args.n_hid,dropout=args.drop_out)
        model_ft = model_ft.to(device)
        optimizer = optim.Adam(model_ft.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        schedular = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestones,gamma=args.gamma)
        criterion = torch.nn.CrossEntropyLoss()
        
        model_ft = train_model_supervised(model_ft, criterion, optimizer, schedular, args.max_epoch, print_freq=args.print_freq)

        
        
