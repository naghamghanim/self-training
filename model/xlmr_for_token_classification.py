from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F
#from TorchCRF import CRF
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1=nn.Linear(hidden_size, hidden_size)
        self.fc2=nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size, hidden_size)
        self.fc4=nn.Linear(hidden_size, hidden_size)
        self.fc5=nn.Linear(hidden_size, 1)

        self.fc1.weight.data.normal_(
            mean=0.0, std=0.03)
        self.fc2.weight.data.normal_(
            mean=0.0, std=0.03)
        self.fc3.weight.data.normal_(
            mean=0.0, std=0.03)

        
    def forward(self, sent_repr):
        d_interm = nn.functional.relu(self.fc1(sent_repr))
        d_interm = nn.functional.relu(self.fc2(d_interm))
        d_interm = nn.functional.relu(self.fc3(d_interm))
        d_interm = nn.functional.relu(self.fc4(d_interm))
        d_logits = self.fc5(d_interm)
        return torch.sigmoid(d_logits)




class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size=768, dropout_p=0.1, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels



        self.label_ignore_idx = label_ignore_idx

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.hidden_size = self.transformer.config.hidden_size
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.classification_head = nn.Linear(self.hidden_size, n_labels)
        #self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        #self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)



        self.device = device

        # initializing classification head
        self.classification_head.weight.data.normal_(
            mean=0.0, std=head_init_range)
        
       
    def forward_generator(self, inputs_ids):
        '''
        Computes a forward pass to generate embeddings 

        Args:
            inputs_ids: tensor of shape (bsz, max_seq_len), pad_idx=1
            labels: temspr pf soze (bsz)

        '''
        transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        #transformer_out, _ = self.model(inputs_ids, features_only=True)
        generator_representation = transformer_out.mean(dim=1) # bsz x hidden
        return generator_representation


    def forward(self, inputs_ids, attention_mask, labels, labels_mask, valid_mask, get_sent_repr= False):
        '''
        Computes a forward pass through the sequence agging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where 
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, sent_repr = self.transformer(
            input_ids=inputs_ids,
            attention_mask=attention_mask
        )


        out_1 = F.relu(self.linear_1(transformer_out))
        out_1 = self.dropout(out_1)
        logits = self.classification_head(out_1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            
            if get_sent_repr:
                return loss, sent_repr
            return loss, sent_repr
        else:
            return logits, sent_repr

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.tokenizer(s)['input_ids']
        # remove <s> and </s> ids
        return tensor_ids[1:-1]

