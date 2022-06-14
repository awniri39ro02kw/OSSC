from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os
import numpy as np
import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Method.utils import myconf
from Method.utils.logger import get_logger
from torchvision.transforms import functional as F
from Method.learning_algo import LearningAlgorithm
from Method.gcn.config import args
from Method.utils.label_utils import reassign_labels, special_train_test_split
import warnings
from scipy.sparse import coo_matrix,csr_matrix
from Method.utils.data import preprocess_adj,preprocess_features

warnings.filterwarnings("ignore")

def meanstd_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    # tensor_norm = np.ones([n_node, n_steps, n_dim])
    tensor_reshape = preprocessing.scale(np.reshape(tensor, [n_node, n_steps * n_dim]), axis=1)
    tensor_norm = np.reshape(tensor_reshape, [n_node, n_steps, n_dim])

    return tensor_norm


def graph_loader(filepath,filename,unseen_list):

    file = np.load(os.path.join(filepath,filename))
    Features  = file['attmats'] #(n_node, n_time, att_dim)
    Labels    = file['labels']  #(n_node, num_classes)
    Graphs    = file['adjs']    #(n_time, n_node, n_node)

    a=1
    supports_list=[]
    features_list=[]
    num_features_nonzero_list=[]
    features_tensor = torch.empty(1, Features.shape[0], Features.shape[2])
    Features_numpy = Features.transpose(1, 0, 2)
    for j in range(Graphs.shape[0]):
        f=Features_numpy[j]
        g=Graphs[j]
        g2=csr_matrix(g)
        f2=csr_matrix(f).tolil()
        features = preprocess_features(f2)
        i = torch.from_numpy(features[0]).long().to("cuda")
        v = torch.from_numpy(features[1]).to("cuda")
        feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to("cuda")
        num_features_nonzero = feature._nnz()
        num_features_nonzero_list.append(num_features_nonzero)

        supports=preprocess_adj(g2)
        i = torch.from_numpy(supports[0]).long().to("cuda")
        v = torch.from_numpy(supports[1]).to("cuda")
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to("cuda")
        supports_list.append(support)

        feature = torch.unsqueeze(feature, 0)
        if j ==0:
            features_tensor=feature
        else:
            features_tensor = torch.cat([features_tensor, feature], dim=0)


    if Graphs.dtype==np.int32:
        Graphs=Graphs.astype(np.float32)
    if Graphs.dtype==np.int64:
        Graphs=Graphs.astype(np.float64)

    #Features = meanstd_normalization_tensor(Features)
    n_node, n_steps, n_dim = np.shape(Features)
    # add self-loop
    for i in range(n_steps):
        Graphs[i, :, :] += np.eye(n_node, dtype=np.float)




    features = F.to_tensor(Features)
    features = features.permute(2,1,0)

    #new  split
    training_rate=0.7
    original_num_classes = Labels.shape[1]
    all_labels = list(range(original_num_classes))
    seen_labels=[]
    for m in all_labels:
        if m not in unseen_list:
            seen_labels.append(m)
    y_true=[]
    for _,i in enumerate(Labels):
        for j in range(len(i)):
            if i[j]==1:
                y_true.append(j)
    y_true=np.array(y_true)

    if unseen :
        y_true = reassign_labels(y_true, seen_labels, -1)


    train_indices, test_valid_indices = special_train_test_split(y_true, unseen_label_index=-1,
                                                                 test_size=1 - training_rate,discard=None)
    test_indices, valid_indices = train_test_split(test_valid_indices, test_size=1.0/ 3.0)

    return train_indices, valid_indices, test_indices, features, supports_list, Labels, y_true,num_features_nonzero_list
                                                    #features_tensor
if __name__ == '__main__':
    l = []

    cfg_file ="./data/dblp5_config.ini"
    config= myconf()
    config.read(cfg_file)

    unseen_list = [] if config.get('Training', 'unseen_list') == '' else [int(i) for i in
                                                                                 config.get('Training',
                                                                                              'unseen_list').split(',')]
    unseen=config.getboolean('Training', 'unseen')

    x_train_idx, x_val_idx, x_test_idx, features, graphs, labels,y_true,num_features_nonzero_list = graph_loader(config.get('User', 'filepath'),
                                                                               config.get('User', 'filename'),unseen_list)

    lambda_1,lambda_2,lambda_3,beta,beta2=[1,100,0.01,0.01,1]

    learning_algo = LearningAlgorithm(1,lambda_1,lambda_2,lambda_3,beta,beta2,cfg_file,num_features_nonzero_list)
    learning_algo.logger.info("dataset:"+learning_algo.filename)
    learning_algo.logger.info("gcn_dropout:" + str(args.dropout))

    log=learning_algo.train_graph(x_train_idx, x_val_idx, x_test_idx, features, graphs, labels,y_true,lambda_1,lambda_2,lambda_3,beta,beta2)

    l.append("\nlambda_1,lambda_2,lambda_3,beta,beta2:[{},{},{},{},{}]\n ".format(lambda_1,lambda_2,lambda_3,beta,beta2)+log+"\n")

    log_file = os.path.join(learning_algo.save_dir, 'best_result.txt')
    logger = get_logger(log_file, 1)
    logger.info("result================================:")
    logger.info("".join(l))
