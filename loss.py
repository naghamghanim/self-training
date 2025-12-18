import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def DANN(features, ad_net):
    """
    :param features: features extracted by the generator (N, hidden_size, H, W)
    :param ad_net: the discriminator network
    :return: loss
    """
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    return nn.BCELoss()(ad_out, dc_target)


def create_matrix(n):
    """
    :param n: matrix size (class num)
    :return a matrix with torch.tensor type:
    for example n=3:
    1     -1/2  -1/2
    -1/2    1   -1/2
    -1/2  -1/2    1
    """
    a = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j:
                a[i,j]=1
            else:
                a[i,j]=-1/(n-1)
    return torch.from_numpy(a).to(device)

def ALDA_loss(ad_out_score, labels_source, slc_labels, slc_probs, source_size, target_size):
    """
    :param ad_out_score: the discriminator output (N, C, H, W)
    :param labels_source: the source ground truth (N, H, W)
    :param softmax_out: the model prediction probability (N, C, H, W)
    :return: 
    adv_loss: adversarial learning loss
    reg_loss: regularization term for the discriminator
    correct_loss: corrected self-training loss
    """
    ad_out = torch.sigmoid(ad_out_score)


    class_num = ad_out.size(1)

    labels_source_mask = torch.zeros(source_size, class_num).to(ad_out.device).scatter_(1, labels_source, 1)
    preds_source_mask = torch.zeros(source_size, class_num).to(ad_out.device).scatter_(1, slc_labels['source'], 1)
    preds_target_mask = torch.zeros(target_size, class_num).to(ad_out.device).scatter_(1, slc_labels['target'], 1)

    # construct the confusion matrix from ad_out. See the paper for more details.
    confusion_matrix = create_matrix(class_num)
    ant_eye = (1-torch.eye(class_num)).to(device).unsqueeze(0)
    confusion_matrix = ant_eye/(class_num-1) + torch.mul(confusion_matrix.unsqueeze(0), ad_out.unsqueeze(1)) #(2*batch_size, class_num, class_num)
    preds_mask = torch.cat([preds_source_mask, preds_target_mask], dim=0) #labels_source_mask
    loss_pred = torch.mul(confusion_matrix, preds_mask.unsqueeze(1)).sum(dim=2)

    # different correction targets for different domains
    loss_target = (1 - preds_target_mask) / (class_num-1)
    loss_target = torch.cat([labels_source_mask, loss_target], dim=0)
    if not ((loss_pred>=0).all() and (loss_pred<=1).all()):
        raise AssertionError
    adv_loss = nn.BCELoss(reduction='none')(loss_pred, loss_target)
    adv_loss = torch.mean(adv_loss)
    
    # reg_loss
    reg_loss = nn.CrossEntropyLoss()(ad_out_score[:source_size], labels_source.view(-1))
    
    # corrected target loss function
    target_probs = 1.0*slc_probs['target']
    correct_target = torch.mul(confusion_matrix.detach()[source_size:], preds_target_mask.unsqueeze(1)).sum(dim=2)
    correct_loss = -torch.mul(target_probs, correct_target)
    correct_loss = torch.mean(correct_loss)
    """
    pickle_object(ad_out,'ad_out')
    pickle_object(preds_mask,'preds_mask')
    pickle_object(preds_source_mask,'preds_source_mask')
    pickle_object(preds_target_mask,'preds_target_mask')
    print(f"adv loss: {adv_loss} -- reg_loss: {reg_loss} -- correct_loss: {correct_loss}")
    raise AssertionError
    """
    return adv_loss, reg_loss, correct_loss

