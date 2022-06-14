import  torch
from    torch import nn
from    torch.nn import functional as F
import numpy as np


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_mask(X_train_idx, X_val_idx, X_test_idx,features, labels):
    X_train_idx=np.sort(X_train_idx)
    X_val_idx=np.sort(X_val_idx)
    X_test_idx=np.sort(X_test_idx)
    idx_test = X_test_idx.tolist()
    idx_train =X_train_idx.tolist()
    idx_val = X_val_idx.tolist()
    train_mask = sample_mask(idx_train, features.shape[1])
    val_mask = sample_mask(idx_val, features.shape[1])
    test_mask = sample_mask(idx_test, features.shape[1])

    x_train = np.zeros(features.shape)
    x_val = np.zeros(features.shape)
    x_test = np.zeros(features.shape)


    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return x_train, x_val, x_test, y_train, y_val, y_test, train_mask, val_mask, test_mask


def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc



def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

