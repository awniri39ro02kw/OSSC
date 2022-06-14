#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings

from scipy.stats import truncnorm
from torch.distributions import laplace

warnings.filterwarnings("ignore")
from torch import nn
import torch
from collections import OrderedDict
from Method.gcn.model import GCN
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from Method.utils.label_utils import reassign_labels
import torch.nn.functional as F


def build_OSSC(cfg, z_dim,num_features_nonzero_list, device='cuda'):
    ### Load parameters for OSSC
    # General
    input_dim = cfg.getint('Network', 'input_dim')
    x_dim = cfg.getint('Network', 'x_dim')
    num_class = cfg.getint('Network', 'num_class')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Deterministic
    dense_x_h = [] if cfg.get('Network', 'dense_x_h') == '' else [int(i) for i in
                                                                  cfg.get('Network', 'dense_x_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    # Inference
    dense_hx_g = [] if cfg.get('Network', 'dense_hx_g') == '' else [int(i) for i in
                                                                    cfg.get('Network', 'dense_hx_g').split(',')]
    dim_RNN_g = cfg.getint('Network', 'dim_RNN_g')
    num_RNN_g = cfg.getint('Network', 'num_RNN_g')
    dense_gz_z = [] if cfg.get('Network', 'dense_gz_z') == '' else [int(i) for i in
                                                                    cfg.get('Network', 'dense_gz_z').split(',')]
    # Prior
    dense_hz_z = [] if cfg.get('Network', 'dense_hz_z') == '' else [int(i) for i in
                                                                    cfg.get('Network', 'dense_hz_z').split(',')]
    # Generation
    dense_hz_x = [] if cfg.get('Network', 'dense_hz_x') == '' else [int(i) for i in
                                                                    cfg.get('Network', 'dense_hz_x').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')


    # Build model
    model =OSSC(num_features_nonzero_list=num_features_nonzero_list,cfg=cfg, input_dim=input_dim, x_dim=x_dim, num_class=num_class, z_dim=z_dim, activation=activation,
                  dense_x_h=dense_x_h,
                  dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                  dense_hx_g=dense_hx_g,
                  dim_RNN_g=dim_RNN_g, num_RNN_g=num_RNN_g,
                  dense_gz_z=dense_gz_z,
                  dense_hz_x=dense_hz_x,
                  dense_hz_z=dense_hz_z,
                  dropout_p=dropout_p, beta=beta, device=device).to(device)
    return model


class OSSC(nn.Module):

    def __init__(self,num_features_nonzero_list, cfg, input_dim, x_dim, num_class, z_dim=16, activation='tanh',
                 dense_x_h=[], dim_RNN_h=128, num_RNN_h=1,
                 dense_hx_g=[], dim_RNN_g=128, num_RNN_g=1,
                 dense_gz_z=[128, 128],
                 dense_hz_z=[128, 128],
                 dense_hz_x=[128, 128],
                 dropout_p=0, beta=1, device='cuda'):

        super().__init__()
        ### General parameters\
        self.trainMode=False
        self.pseudo_index_threshold=[]
        self.pseudo_index_loss=[]
        self.epoch=2
        self.num_features_nonzero_list=num_features_nonzero_list
        self.input_dim = input_dim
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Deterministic RNN (forward)
        self.dense_x_h = dense_x_h
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        ### Inference
        self.dense_hx_g = dense_hx_g
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.dense_gz_z = dense_gz_z
        ### Generation z
        self.dense_hz_z = dense_hz_z
        ### Generation x
        self.dense_hz_x = dense_hz_x
        ### Beta-loss
        self.beta = beta
        self.num_class = num_class
        self.n_time = cfg.getint('Network', 'n_time')
        self.unseen = cfg.getboolean("Training", "unseen")
        self.unseen_num = cfg.getint("Training", "unseen_num")
        self.unseen_label_index = -1
        self.unseen_list = [] if cfg.get('Training', 'unseen_list') == '' else [int(i) for i in
                                                                                cfg.get('Training',
                                                                                        'unseen_list').split(
                                                                                    ',')]
        self.build()

    def build(self):

        self.gcn = GCN(self.input_dim, self.x_dim, num_features_nonzero=False).to(device=self.device)

        #######################
        #### Deterministic ####
        #######################
        # 1. x_tm1 -> h_t

        dic_layers = OrderedDict()
        if len(self.dense_x_h) == 0:
            dim_x_h = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_h = self.dense_x_h[-1]
            for n in range(len(self.dense_x_h)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.x_dim, self.dense_x_h[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_x_h[n - 1], self.dense_x_h[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_h = nn.Sequential(dic_layers)

        # 2. h_t, forward recurrence
        self.rnn_h = nn.LSTM(dim_x_h, self.dim_RNN_h, self.num_RNN_h)

        ###################
        #### Inference ####
        ###################
        # 1. h_t x_t -> g_t
        dic_layers = OrderedDict()
        if len(self.dense_hx_g) == 0:
            dim_hx_g = self.dim_RNN_h + self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_g = self.dense_hx_g[-1]
            for n in range(len(self.dense_hx_g)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dim_RNN_h + self.x_dim, self.dense_hx_g[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_hx_g[n - 1], self.dense_hx_g[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_g = nn.Sequential(dic_layers)

        # 2. g_t, backward recurrence
        self.rnn_g = nn.LSTM(dim_hx_g, self.dim_RNN_g, self.num_RNN_g)

        # 3. g_t z_tm1 -> z_t, inference
        dic_layers = OrderedDict()
        if len(self.dense_gz_z) == 0:
            dim_gz_z = self.dim_RNN_g + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_gz_z = self.dense_gz_z[-1]
            for n in range(len(self.dense_gz_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dim_RNN_g + self.z_dim, self.dense_gz_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_gz_z[n - 1], self.dense_gz_z[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_gz_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_gz_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_gz_z, self.z_dim)

        ######################
        #### Generation z ####
        ######################
        # 1. h_t z_tm1 -> z_t
        dic_layers = OrderedDict()
        if len(self.dense_hz_z) == 0:
            dim_hz_z = self.dim_RNN_h + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_z = self.dense_hz_z[-1]
            for n in range(len(self.dense_hz_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dim_RNN_h + self.z_dim, self.dense_hz_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_hz_z[n - 1], self.dense_hz_z[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_z = nn.Sequential(dic_layers)
        self.prior_mean = nn.Linear(dim_hz_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_hz_z, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        # 1. h_t z_t -> x_t
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN_h + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dim_RNN_h + self.z_dim, self.dense_hz_x[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_hz_x[n - 1], self.dense_hz_x[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_hz_x, self.y_dim)

        ######################
        #### classification ####
        ######################
        if self.unseen == False:
            self.classf = nn.Sequential(nn.Linear(self.z_dim, self.num_class),
                                        nn.Softmax())
        else:
            self.classf = nn.Sequential(nn.Linear(self.z_dim, self.num_class - self.unseen_num),
                                            nn.Softmax())

    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        """"trunc laplace """
        eps = torch.randn_like(std)
        return torch.addcmul(mean, eps, std)

    def deterministic_h(self, x_tm1):

        x_h = self.mlp_x_h(x_tm1)
        h, _ = self.rnn_h(x_h)

        return h


    def inference(self, x, h, graphs):

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed 11
        z_mean = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_logvar = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)

        # 1. From h_t and x_t to g_t
        hx_g = torch.cat((h, x), -1)

        hx_g = self.mlp_hx_g(hx_g)
        g_inverse, _ = self.rnn_g(torch.flip(hx_g, [0]))
        g = torch.flip(g_inverse, [0])

        # 2. From g_t and z_tm1 to z_t
        for t in range(seq_len):
            gz_z = torch.cat((g[t, :, :], z_t), -1)
            #support = graphs[t, :, :]
            support = graphs[t]
            gz_z = self.mlp_gz_z(gz_z)
            z_mean[t, :, :] = self.inf_mean(gz_z)
            z_logvar[t, :, :] = self.inf_logvar(gz_z)
            z_t = self.reparameterization(z_mean[t, :, :], z_logvar[t, :, :])
            z[t, :, :] = z_t

        return z, z_mean, z_logvar

    def generation_z(self, h, z_tm1, seq_len, batch_size, graphs, filter_unseen=False):
        if filter_unseen:
            sampling_time = 100
        else:
            sampling_time = 1
        # sampling_time = 1
        hz_z = torch.cat((h, z_tm1), -1)
        hz_z = self.mlp_hz_z(hz_z)
        # hz_z = self.mlp_hz_z2(hz_z)
        z_mean_p = self.prior_mean(hz_z)
        z_logvar_p = self.prior_logvar(hz_z)

        z_p = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        # for t in range(seq_len):
        t = -1
        z_p_t = self.reparameterization(z_mean_p[t, :, :], z_logvar_p[t, :, :])
        for _ in range(1, sampling_time):
            z_p_t += self.reparameterization(z_mean_p[t, :, :], z_logvar_p[t, :, :])
        z_p_t /= sampling_time

        z_p[t, :, :] = z_p_t

        return z_p, z_mean_p, z_logvar_p

    def generation_x(self, z, h, graphs):

        # 1. z_t and h_t to y_t
        hz_x = torch.cat((h, z), -1)
        hz_x = self.mlp_hz_x(hz_x)
        log_y = self.gen_logvar(hz_x)
        y = torch.exp(log_y)
        return y


    def forward(self,y_true, labels, features, graphs, masks, label_ori, lambda_1, lambda_2, lambda_3, beta, beta2, valid_mask,
                filter_unseen=False, threshold=None):

        timesteps = features.shape[0]
        emb_gcn = torch.empty(1, features.shape[1], self.x_dim)  # 16 - embedding size
        emb_gcn = emb_gcn.to(self.device)
        for i in range(timesteps):
            feature = features[i]
            #graph = graphs[i, :, :]
            graph = graphs[i]
            out = self.gcn((feature, graph))

            emb_gcn_l1 = out[0]

            emb_gcn_l1 = torch.unsqueeze(emb_gcn_l1, 0)
            emb_gcn = torch.cat([emb_gcn, emb_gcn_l1], dim=0)

        emb_gcn = emb_gcn[torch.arange(emb_gcn.size(0)) != 0]
        emb_gcn = emb_gcn.permute(1, 2, 0)
        x = emb_gcn

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(-1, 0, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]

        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)  # x here represents u in the paper
        x_tm1 = torch.cat((x_0, x[:-1, :, :]), 0)


        ##second layer   257->(linear)128->(lstm)64
        h = self.deterministic_h(x_tm1)

        ##third layer
        z, z_mean, z_logvar = self.inference(x, h, graphs)


        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat((z_0, z[:-1, :, :]), 0)

        ##fourth layer
        z_p, z_mean_p, z_logvar_p = self.generation_z(h, z_tm1, seq_len, batch_size, graphs, filter_unseen)
        y = self.generation_x(z, h, graphs)

        # softmax layer for classification
        z_p_last = z_p[-1, :, :]  # only use the last time step's embedding
        z_p_last = z_p_last.squeeze()
        label = self.classf(z_p_last)

        if filter_unseen:

            loss_tot, cross_entropy, class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc, auc_mac, auc_mic, f1_mac, f1_mic, threshold_ = self.get_loss(
                y_true,labels, x, y, graphs, z, z_mean, z_logvar,
                z_mean_p, z_logvar_p,
                seq_len, batch_size, masks, label, label_ori, lambda_1, lambda_2, lambda_3, beta, beta2, valid_mask,
                filter_unseen, threshold)
            self.loss = (
                loss_tot, cross_entropy, class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class,
                acc, auc_mac, auc_mic, f1_mac, f1_mic, threshold_)
        else:

            loss_tot, cross_entropy, class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc, auc_mac, auc_mic, f1_mac, f1_mic = self.get_loss(
                y_true,labels, x, y, graphs, z, z_mean, z_logvar,
                z_mean_p, z_logvar_p,
                seq_len, batch_size, masks, label, label_ori, lambda_1, lambda_2, lambda_3, beta, beta2, valid_mask,
                filter_unseen, threshold)
            self.loss = (
            loss_tot, cross_entropy, class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc,
            auc_mac, auc_mic, f1_mac, f1_mic)
        if self.trainMode:
            assert ~torch.isnan(loss_tot)

        return

    def get_loss(self,y_true, labels, x, y, graphs_ori, z, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len, batch_size,
                 mask, label_pre, label_ori, lambda_1, lambda_2, lambda_3, beta, beta2, valid_mask, filter_unseen,
                 threshold):
        global all_acc
        soft = nn.Softmax(dim=1)
        #adj = soft(graphs_ori[-1, :, :])
        adj = soft(graphs_ori[-1].to_dense())
        adj = torch.where(adj < 1e-6, torch.full_like(adj, 1e-6), adj)

        adj_p = torch.bmm(z, z.permute(0, 2, 1))
        adj_p = soft(adj_p[-1, :, :])
        adj_p = torch.where(adj_p < 1e-6, torch.full_like(adj_p, 1e-6), adj_p)

        masks = torch.zeros(adj.shape)
        masks[mask, :] = 1
        masks = masks.to(self.device)
        t1 = adj / adj_p
        t1 = torch.mul(t1, masks)
        t2 = torch.log(adj / adj_p)
        t2 = torch.mul(t2, masks)

        loss_recon_adj = torch.sum(t1 - t2 - 1)
        loss_recon_adj = loss_recon_adj / (batch_size * seq_len)

        # loss of reconstraction of feature matrics
        x = torch.where(x == 0, torch.full_like(x, 1e-6), x)

        masks = torch.zeros(x.shape)
        masks[:, mask, :] = 1
        masks = masks.to(self.device)
        t1 = x / y
        t1 = torch.mul(t1, masks)
        t2 = torch.log(x / y)
        t2 = torch.mul(t2, masks)
        loss_recon = torch.sum(t1 - t2 - 1)

        # step 3.2 : add classification loss
        label_pre_ = label_pre
        label_ori = label_ori[mask == True, :]
        label_pre = label_pre[mask == True, :]

        if type(label_ori) == np.ndarray:
            label_ori = torch.from_numpy(label_ori)
            label_ori = label_ori.to(self.device)
        if type(label_pre) == np.ndarray:
            label_pre = torch.from_numpy(label_pre)
            label_pre = label_pre.to(self.device)
        if type(labels) == np.ndarray:
            labels = torch.from_numpy(labels)
            labels = labels.to(self.device)
        label_pre_maxv, label_pre_maxi = torch.max(label_pre, dim=1)
        y_pre = label_pre_maxi.detach().cpu().numpy()
        probs = label_pre_maxv.detach().cpu().numpy()
        y_ori=y_true[mask.detach().cpu()==1]

        if filter_unseen:
            if threshold is None:
                valid_pre = label_pre.detach().cpu().numpy()[y_ori != self.unseen_label_index]
                probs = probs[y_ori != self.unseen_label_index]
                valid_all = list(range(len(valid_pre)))
                if self.epoch%100==0:
                    self.pseudo_index_threshold=[]
                    entropy_rate = 0.05
                    entropy_value = []
                    for i, x in enumerate(valid_pre):
                        entropy_value.append((i, self.calc_ent(x)))
                    entropy_value.sort(key=lambda x: x[1], reverse=True)
                    pseudo_unseen_num = int(entropy_rate * len(entropy_value))
                    i = 0
                    while i < pseudo_unseen_num:
                        self.pseudo_index_threshold.append(entropy_value[i][0])
                        i += 1
                valid_seen=list(set(valid_all).difference(set(self.pseudo_index_threshold)))
                threshold = (probs[valid_seen].mean() + probs[self.pseudo_index_threshold].mean()) / 2.0
            else:
                y_pre[probs < threshold] = self.unseen_label_index



        acc = accuracy_score(y_ori, y_pre)

        l_ori = label_ori.cpu().detach().numpy()
        l_pre = label_pre.cpu().detach().numpy()

        if self.unseen:
            auc_mac = roc_auc_score(l_ori[:, :l_pre.shape[1]], l_pre, average='macro', multi_class='ovo')
            auc_mic = roc_auc_score(l_ori[:, :l_pre.shape[1]], l_pre, average='micro', multi_class='ovo')
        else:
            auc_mac = roc_auc_score(l_ori, l_pre, average='macro', multi_class='ovo')
            auc_mic = roc_auc_score(l_ori, l_pre, average='micro', multi_class='ovo')

        # F1
        f1_mac = f1_score(y_ori, y_pre, average='macro')
        f1_mic = f1_score(y_ori, y_pre, average='micro')
        class_uncertainty_loss=0
        # compute Label loss and class uncertainty loss
        if self.unseen:
            labels_i = torch.tensor(y_ori)
            labels_i = labels_i.to(self.device)

            if label_ori.shape[1] > (self.num_class - len(self.unseen_list)):
                seen_labels = list(range(self.num_class-self.unseen_num))
                tt = reassign_labels(y_ori, seen_labels, self.num_class - len(self.unseen_list))
                labels_i = torch.tensor(tt)
                labels_i = labels_i.to(self.device)
                label_one_hot = F.one_hot(labels_i, num_classes=(self.num_class - len(self.unseen_list) + 1))
                label_one_hot = label_one_hot[:, :self.num_class - len(self.unseen_list)]
            else:
                label_one_hot = F.one_hot(labels_i, num_classes=(self.num_class - len(self.unseen_list)))
            cross_entropy = -torch.sum(label_one_hot * torch.log(label_pre))


            val_label_pre = label_pre_[valid_mask == True, :]
            ######
            if self.epoch%100==0 and self.trainMode:
                entropy_rate=0.05
                train_pre=label_pre_[valid_mask == True, :].cpu().detach().numpy()
                entropy_value=[]
                for i,x in enumerate(train_pre):
                    entropy_value.append((i,self.calc_ent(x)))
                entropy_value.sort(key=lambda x:x[1],reverse=True)
                pseudo_unseen_num=int(entropy_rate*len(entropy_value))
                self.pseudo_index_loss=[]
                i=0
                while i<pseudo_unseen_num:
                    self.pseudo_index_loss.append(entropy_value[i][0])
                    i+=1
            val_label_pre=val_label_pre[self.pseudo_index_loss]




            ##########
            val_label_pre = torch.clamp(val_label_pre, 1e-7, 1.0)
            _v, ind = torch.max(val_label_pre, dim=1)
            list_v = _v.cpu().detach().numpy().tolist()
            list_v.sort()
            c = []
            a = list_v[list_v.__len__() // 10]
            b = list_v[list_v.__len__() - list_v.__len__() // 10]
            list_v = _v.cpu().detach().numpy().tolist()
            for i in range(len(list_v)):
                if list_v[i] > a and list_v[i] < b:
                    c.append(True)
                else:
                    c.append(False)

            c = torch.tensor(c)
            val_label_pre = val_label_pre[c]
            ##########


            if self.trainMode:
                class_uncertainty_loss = torch.mean(val_label_pre * torch.log(val_label_pre))


        else:
            _, labels_i = torch.max(labels, dim=1)
            labels_i = torch.tensor(y_true)
            labels_i = labels_i.to(self.device)
            label_one_hot = F.one_hot(labels_i, num_classes=self.num_class)
            cross_entropy = -torch.sum(label_one_hot * torch.log(label_pre))

            class_uncertainty_loss = 0

        loss_class = 0

        if self.trainMode:


            loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar_p
                                        - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))

            loss_recon = loss_recon / (batch_size * seq_len)
            loss_KLD = loss_KLD / (batch_size * seq_len)
            loss_tot = lambda_1 * loss_recon_adj + lambda_2 * loss_recon + lambda_3 * loss_KLD + beta * cross_entropy + beta2 * class_uncertainty_loss
        else:
            loss_KLD=0
            loss_tot=0

        if filter_unseen:
            return loss_tot, cross_entropy, class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc, auc_mac, auc_mic, f1_mac, f1_mic, threshold
        else:
            return loss_tot, cross_entropy, class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc, auc_mac, auc_mic, f1_mac, f1_mic

    def calc_ent(self,x):
        ent = 0.0
        for p in x:
            logp = np.log2(p)
            ent -= p * logp
        return ent

    def get_info(self):

        info = []
        info.append('----- Inference -----')
        info.append('>>>> From x_tm1 to h_t')
        for layer in self.mlp_x_h:
            info.append(str(layer))
        info.append('>>>> Forward RNN to generate h_t')
        info.append(str(self.rnn_h))
        info.append('>>>> From h_t and x_t to g_t')
        for layer in self.mlp_hx_g:
            info.append(str(layer))
        info.append('>>>> Backward RNN to generate g_t')
        info.append(str(self.rnn_g))
        info.append('>>>> From z_tm1 and g_t to z_t')
        for layer in self.mlp_gz_z:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append('mean: ' + str(self.inf_mean))
        info.append('logvar: ' + str(self.inf_logvar))

        info.append('----- Generation x -----')
        info.append('>>>> From h_t and z_t to x_t')
        for layer in self.mlp_hz_x:
            info.append(str(layer))
        info.append(str(self.gen_logvar))

        info.append('----- Generation z -----')
        info.append('>>>> From h_t and z_tm1 to z_t')
        for layer in self.mlp_hz_z:
            info.append(str(layer))
        info.append('prior mean: ' + str(self.prior_mean))
        info.append('prior logvar: ' + str(self.prior_logvar))

        return info
