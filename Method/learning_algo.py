#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
import random
from torch.utils.data import SubsetRandomSampler

warnings.filterwarnings("ignore")
import os
import shutil
import socket
import datetime
from Method.utils.logger import get_logger
from .utils.read_config import myconf
from Method.gcn.config import args
from Method.model import build_OSSC
from Method.gcn.utils import *


class LearningAlgorithm():


    def __init__(self,count,lambda_1,lambda_2,lambda_3,beta,beta2,config_file,num_features_nonzero_list):


        # Load config parser
        self.config_file = config_file
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')


        self.count=count
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.model_name = self.cfg.get('Network', 'name')
        self.unseen_list=[] if self.cfg.get('Training', 'unseen_list') == '' else [int(i) for i in self.cfg.get('Training', 'unseen_list').split(',')]
        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        self.z_dim = self.cfg.getint('Network', 'z_dim')

        self.best_train_acc=0
        self.best_train_epoch=-1
        self.best_val_acc = 0
        self.best_test_acc=0
        self.best_val_epoch = -1
        self.best_test_epoch=-1
        self.best_auc_mic=0
        self.best_auc_mac= 0
        self.best_auc_mic_epoch=-1
        self.best_auc_mac_epoch=-1
        self.best_f1_mic=0
        self.best_f1_mac= 0
        self.best_f1_mic_epoch=-1
        self.best_f1_mac_epoch=-1
        self.best_state_dict = None
        self.num_class=self.cfg.getint('Network', 'num_class')
        self.unseen_num=len(self.unseen_list)
        self.filename=self.cfg.get('User', 'filename')
        self.filepath=self.cfg.get('User', 'filepath')
        saved_root = self.cfg.get('User', 'saved_root')
        filename = "{}_count_{}_{}".format(self.filename,self.count,self.date)
        self.save_dir = os.path.join(saved_root, filename)

        saved_root = self.cfg.get('User', 'saved_root')
        self.unseen=self.cfg.getboolean('Training', 'unseen')
        self.save_dir = os.path.join(saved_root, filename)
        if not (os.path.isdir(self.save_dir)):
            os.makedirs(self.save_dir)

        # Save the model configuration
        save_cfg = os.path.join(self.save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(self.save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        self.logger = get_logger(log_file, logger_type)


        cudastatus=self.cfg.get('Training', 'use_cuda')
        if cudastatus=='True':
                        self.device = 'cuda'
        else:
            self.device='cpu'
        # Build model
        self.model = build_OSSC(cfg=self.cfg,z_dim=self.z_dim,num_features_nonzero_list=num_features_nonzero_list, device=self.device)


    def init_optimizer(self):

        # Load 
        self.optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        # Init optimizer (Adam by default)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def get_basic_info(self):

        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))

        return basic_info

    def train_graph(self,  x_train_idx, x_val_idx, x_test_idx, features, graphs, labels,y_true,lambda_1,lambda_2,lambda_3,beta,beta2):

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Print basical infomation
        for log in self.get_basic_info():
            self.logger.info(log)
        self.logger.info('In this experiment, result will be saved in: ' + self.save_dir)

        # Init optimizer
        self.init_optimizer()


        train_num = x_train_idx.shape[0]
        val_num = x_val_idx.shape[0]
        test_num=x_test_idx.shape[0]
        log_message = 'Training samples: {}'.format(train_num)
        self.logger.info(log_message)
        log_message = 'Validation samples: {}'.format(val_num)
        self.logger.info(log_message)
        log_message = 'Test samples: {}'.format(test_num)
        self.logger.info(log_message)


        train_dataset, val_dataset, test_dataset, train_label, val_label, test_label, \
        train_mask, val_mask, test_mask = get_mask(x_train_idx, x_val_idx, x_test_idx, features, labels)

        if self.unseen:
            train_label = np.delete(labels, self.unseen_list, 1)

        batch_size=self.cfg.getint('Network','n_time')


        # train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
        # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
        # test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False )

        train_mask = torch.from_numpy(train_mask.astype(np.int)).to(self.device)
        train_label = torch.from_numpy(train_label).long().to(self.device)
        val_mask = torch.from_numpy(val_mask.astype(np.int)).to(self.device)
        val_label = torch.from_numpy(labels).long().to(self.device)
        test_mask = torch.from_numpy(test_mask.astype(np.int)).to(self.device)
        test_label = torch.from_numpy(labels).long().to(self.device)
        #graphs = graphs.to(self.device)
        a=1
        features=features.to(self.device)


        return self.train_normal(labels,y_true,self.logger, self.save_dir, features, features,features, train_mask, val_mask, test_mask,train_num, val_num,test_num, graphs, train_label, val_label,test_label,lambda_1,lambda_2,lambda_3,beta,beta2)



    def train_normal(self,labels,y_true, logger, save_dir, train_dataloader, val_dataloader,test_dataloader,  train_mask, val_mask,test_mask, train_num, val_num,test_num, graphs, train_label, val_label,test_label,lambda_1,lambda_2,lambda_3,beta,beta2):

        # Load training parameters
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        print(self.model)
        log_message = 'model structure: {}'.format(self.model)
        logger.info(log_message)

        # Create python list for loss
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        test_loss= np.zeros((epochs,))
        train_recon = np.zeros((epochs,))
        train_KLD = np.zeros((epochs,))

        val_recon = np.zeros((epochs,))
        val_KLD = np.zeros((epochs,))
        test_recon = np.zeros((epochs,))
        test_KLD = np.zeros((epochs,))
        cpt_patience = 0

        # Define optimizer (might use different training schedule)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        log_message = 'optimizer: {}'.format(self.optimizer)
        logger.info(log_message)
        # Train with mini-batch SGD
        print('epochs: ')
        print(epochs)
        log_message = 'epochs: {}'.format(epochs)
        logger.info(log_message)

        m = -1
        begin_time = datetime.datetime.now()
        for epoch in range(epochs):
            self.model.epoch=epoch
            start_time = datetime.datetime.now()

            # Training

                #batch_data = batch_data.to(self.device)
            self.model.trainMode=True
            recon_batch_data = self.model(y_true,labels,train_dataloader, graphs, train_mask, train_label,lambda_1,lambda_2,lambda_3,beta,beta2,valid_mask=train_mask )

            loss_tot,cross_entropy,class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc,auc1,auc2,f1mac,f1mic= self.model.loss
            optimizer.zero_grad()
            if acc>self.best_train_acc:
                self.best_train_acc=acc
                self.best_train_epoch=epoch
            m = m + 1
            log_message = 'trian_loss %d: loss_tot: %3f, ' \
                          'cross_entropy: %3f ,class_uncertainty_loss: %3f ,' \
                          'loss_recon: %3f, loss_KLD: %3f,' \
                          ' loss_recon_adj: %3f, loss_calss: %3f,\n ' \
                          'acc: %3f,auc_mac:%3f,auc_mic:%3f,f1_mac:%3f,f1_mic:%3f                    ' \
                          ' \n best_train_acc: %3f,at epoch:%d'%(m,
                        loss_tot,cross_entropy,class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class,acc,auc1,auc2,f1mac,f1mic,self.best_train_acc,self.best_train_epoch)


            loss_tot.backward(retain_graph=True)

            optimizer.step()



            train_loss[epoch] += loss_tot.item()
            train_recon[epoch] += loss_recon.item()
            train_KLD[epoch] += loss_KLD.item()
            logger.info(log_message)

            end_time = datetime.datetime.now()
            t=str(end_time - start_time)
            logger.info('train time for one epoch='+t)

            # Validation


            self.model.trainMode=False
            recon_batch_data = self.model(y_true,labels,val_dataloader, graphs, val_mask, val_label, lambda_1,lambda_2,lambda_3,beta,beta2,valid_mask=1-val_mask,filter_unseen=True)

            loss_tot,cross_entropy,class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc,auc1,auc2,f1mac,f1mic,threshold = self.model.loss
            if acc>self.best_val_acc:
                self.best_val_acc=acc
                self.best_val_epoch=epoch


            # Test

            recon_batch_data = self.model(y_true,labels,test_dataloader, graphs, test_mask, test_label,lambda_1,lambda_2,lambda_3,beta2,beta,valid_mask=1-test_mask,filter_unseen=True,threshold=threshold)

            loss_tot,cross_entropy,class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj, loss_class, acc, auc_mac, auc_mic, f1mac, f1mic,_ = self.model.loss
            if acc > self.best_test_acc:#1
                if acc - self.best_test_acc > 0.0005:
                    stop_time = datetime.datetime.now()
                    converge_epoch=epoch
                cpt_patience = 0
                self.best_test_acc = acc
                self.best_test_epoch = epoch
                self.best_state_dict=self.model.state_dict()

            else:
                cpt_patience +=1
            if auc_mac>self.best_auc_mac:
                self.best_auc_mac=auc_mac
                self.best_auc_mac_epoch=epoch
            if auc_mic>self.best_auc_mic:
                self.best_auc_mic=auc_mic
                self.best_auc_mic_epoch=epoch
            if f1mac>self.best_f1_mac:
                self.best_f1_mac=f1mac
                self.best_f1_mac_epoch=epoch
            if f1mic>self.best_f1_mic:
                self.best_f1_mic=f1mic
                self.best_f1_mic_epoch=epoch


            log_message = 'test at epoch %d:  acc: %3f,auc_mac:%3f,auc_mic:%3f,f1_mac:%3f,f1_mic:%3f  \n ' \
                      'best_test_acc: %3f,at epoch:%d \n ' \
                      'best_auc_mac: %3f,at epoch:%d \n' \
                      'best_auc_mic: %3f,at epoch:%d \n' \
                      'best_f1_mac: %3f,at epoch:%d \n' \
                      'best_f1_mic: %3f,at epoch:%d\n' \
                      'converge at epoch:%d\n' % (m,acc, auc1, auc2, f1mac, f1mic,
                                                         self.best_test_acc, self.best_test_epoch,self.best_auc_mac,self.best_auc_mac_epoch,
                                                         self.best_auc_mic,self.best_auc_mic_epoch,self.best_f1_mac,self.best_f1_mac_epoch,
                                                         self.best_f1_mic,self.best_f1_mic_epoch,converge_epoch)
                                                         # loss_tot,cross_entropy,class_uncertainty_loss, loss_recon, loss_KLD, loss_recon_adj,
                                                         # loss_class,


            #print(log_message)
            logger.info(log_message)



            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            train_recon[epoch] = train_recon[epoch] / train_num
            train_KLD[epoch] = train_KLD[epoch]/ train_num


            # Training time
            end_time = datetime.datetime.now()
            t=str(end_time - start_time)
            log_message = 'Epoch: {} train loss: {:.4f} val loss {:.4f} '.format(epoch, train_loss[epoch], val_loss[epoch])#, interval)
            logger.info(log_message)
            logger.info('total time for one epoch='+t)
            logger.info("================================")


            # Stop traning if early-stop triggers
            if cpt_patience==early_stop_patience:
                torch.save(self.best_state_dict,self.save_dir+"/best_state.pt")
                logger.info('Early stop patience achieved')
                break

        whole_time=str(stop_time-begin_time)




        log_message =  'z_dim:%d\n'     \
                        'count:%d \n' \
                      'best_test_acc: %3f,at epoch:%d \n' \
                      'best_auc_mac: %3f,at epoch:%d \n' \
                      'best_auc_mic: %3f,at epoch:%d \n' \
                      'best_f1_mac: %3f,at epoch:%d \n' \
                      'best_f1_mic: %3f,at epoch:%d \n' \
                       'converge at epoch:%d \n' \
                       % (self.z_dim,self.count,self.best_test_acc, self.best_test_epoch, self.best_auc_mac,
                                                           self.best_auc_mac_epoch,
                                                           self.best_auc_mic, self.best_auc_mic_epoch, self.best_f1_mac,
                                                           self.best_f1_mac_epoch,
                                                           self.best_f1_mic, self.best_f1_mic_epoch,converge_epoch)

        log_message+=('total time to converge: '+whole_time)
        return log_message