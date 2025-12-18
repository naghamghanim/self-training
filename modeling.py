import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import BertModel

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

def pickle_object(obj,filename):
    outfile = open(filename,'wb')
    pickle.dump(obj,outfile)
    outfile.close()


class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))
        a = F.softmax(self.contx(u), dim=1)
        s = (a * inp).sum(1)
        return s
        
class BertLayer(nn.Module):
    def __init__(self, dropout_prob=0.15,
                 pretrained_path='aubmindlab/bert-base-arabert', lr_mult= 1):
        super(BertLayer, self).__init__()

        self.transformer = BertModel.from_pretrained(pretrained_path)
        self.lr_mult = lr_mult
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.attention2 = AttentionWithContext(self.transformer.config.hidden_size)
        self.attention3 = AttentionWithContext(self.transformer.config.hidden_size)


    def forward(self, input_ids=None, attention_mask=None, is_train=True,valid_mask=None, segments = None, segments_mask=None,  segments_indices_mask = None, with_segs = True, is_source = True):

        transformer_out, sent_repr = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,return_dict=False
        )
        #pickle_object(transformer_out,"transformer_out.pickle")
        #pickle_object(segments,"segments.pickle")
        #pickle_object(segments_mask,"segments_mask.pickle")
        #pickle_object(segments_indices_mask,"segments_indices_mask.pickle")
        transformer_out = self.dropout(transformer_out)
        sent_repr = self.dropout2(sent_repr)
        out_vectors = []
        for i, sent_out in enumerate(transformer_out):
            current_vec = []
            selected_segments = segments[i][segments_mask[i]]
            for j, seg in enumerate(selected_segments):
                selected_segment_indices = seg[segments_indices_mask[i][j]]
                #current_vec.append(sent_out[selected_segment_indices].mean(dim=0))
                if with_segs:
                    current_vec.append(sent_out[selected_segment_indices].max(dim=0).values)
                else:
                    current_vec.append(sent_out[selected_segment_indices][0])
                #pickle_object(sent_out[selected_segment_indices], "sent_out_selected_segments.pickle")
            while len(current_vec) < 128:
                current_vec.append(sent_out[-2])
            out_vectors.append(torch.stack(current_vec))
        #print(out_vectors[0])
        #pickle_object(out_vectors,"out_vectors.pickle")
        if  valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)
            sum_mask_embeddings = valid_mask.sum(dim=1)
            mask_embeddings = transformer_out * mask.float()
            sent_repr = mask_embeddings.sum(1) / sum_mask_embeddings.unsqueeze(1)
        
        #print(torch.stack(out_vectors).shape)
        #print(transformer_out.shape)
        if is_source:
            sent_repr = self.attention2(transformer_out)
        else:
            sent_repr = self.attention3(transformer_out)

        return torch.stack(out_vectors), sent_repr

    def get_parameters(self):
        return [{"params": self.parameters(),"lr_mult":1, 'decay_mult':1}]

    def output_num(self):
        return self.transformer.config.hidden_size

class TokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size=768, dropout_p=0.1, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels

        self.label_ignore_idx = label_ignore_idx

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.hidden_size = self.transformer.config.hidden_size
        #self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.classification_head = nn.Linear(self.hidden_size, n_labels)
        #self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        #self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)

        self.device = device

        # initializing classification head
        self.classification_head.weight.data.normal_(
            mean=0.0, std=head_init_range)

    def forward(self, x , labels, segments_mask = None, get_sent_repr= False):

        out_1 = self.dropout(x)
        logits = self.classification_head(out_1)
        #print(logits)
        #pickle_object(out_1,"outs_valid.pickle")
        #pickle_object(logits,"logits_valid_model.pickle")

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            #print(labels.view(-1)[segments_mask.view(-1)])
            #print(labels.view(-1)[segments_mask.view(-1)].shape)
            #print(logits.view(-1, self.n_labels).shape)
            '''print("logits:", logits.shape)                 # [B, L, C] usually
            print("labels:", labels.shape)                 # [B, L]
            print("segments_mask:", segments_mask.shape)   # [B, L]
            print("segments_mask dtype:", segments_mask.dtype)
            print("B,L from logits:", logits.size(0), logits.size(1))
            print("mask numel:", segments_mask.numel(), "logits positions:", logits.size(0)*logits.size(1))
            #logits.view(-1, logits.shape[-1]), gold_tags.view(-1)'''
            loss = loss_fct(
                logits.view(-1, self.n_labels)[segments_mask.view(-1)], labels.view(-1)[segments_mask.view(-1)])
            
            if get_sent_repr:
                return loss
            return loss
        else:
            #pickle_object(logits,"outs_valid.pickle")
            #print("I am here")
            
            #print("----------------------------------")
            
            return logits

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.tokenizer(s)['input_ids']
        # remove <s> and </s> ids
        return tensor_ids[1:-1]

    def get_parameters(self):
        return [{"params": self.parameters(),"lr_mult":1, 'decay_mult':1}]


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, max_iter=10000):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]

class Multi_AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, class_num, max_iter=10000.0):
        super(Multi_AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x, grl=True):
        if self.training:
            self.iter_num += 1
        if grl and self.training:
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
            x = x * 1.0
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def scale_hook(coeff):
    def fun1(grad):
        return coeff * grad.clone()

    return fun1



def mcc_loss(outputs_target, u_segments_mask, temperature=4, class_num=2):
    #print(f"number of classes is {class_num}")
    #pickle_object(outputs_target,'outputs_target.pickle')
    outputs_target =  outputs_target.view(-1,class_num)[u_segments_mask.view(-1)]
    train_bs = outputs_target.size(0)
    outputs_target_temp = outputs_target / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(target_softmax_out_temp)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss

def EntropyLoss(input_,u_segments_mask, class_num=2):
    # print("input_ shape", input_.shape)
    input_ =  input_.view(-1,class_num)[u_segments_mask.view(-1)]
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out + 1e-5)))
    return entropy / float(input_.size(0))

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
