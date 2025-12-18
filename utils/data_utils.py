import os
import logging

import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader


class ReviewDataset(Dataset):
    def __init__(self, df, pretraine_path='aubmindlab/bert-base-arabertv2', max_length=128):
        self.df = df

    def __getitem__(self, index):

        input_ids = torch.tensor(self.df[index].input_ids, dtype=torch.long)
        attention_mask = torch.tensor(self.df[index].input_mask, dtype=torch.long)
        labels_ids = torch.tensor(self.df[index].labels, dtype=torch.long)
        #segments = [torch.tensor(seg, dtype=torch.long) for seg in self.df[index].segments]
        #segments_indices_mask = [torch.tensor(seg, dtype=torch.long) for seg in self.df[index].segments_indices_mask]
        segments  = torch.tensor(self.df[index].segments)
        segments_indices_mask = torch.tensor(self.df[index].segments_indices_mask, dtype=torch.bool)
        segments_mask = torch.tensor(self.df[index].segments_mask, dtype=torch.bool)
                
        return input_ids, attention_mask, labels_ids, segments, segments_mask, segments_indices_mask
    
    def __len__(self):
        return len(self.df)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids=None, labels=None, input_mask=None, segments=None, segments_mask=None, segments_indices_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.labels = labels
        self.segments = segments
        self.segments_mask = segments_mask
        self.segments_indices_mask = segments_indices_mask

class SequenceClassificationProcessor:
   
    def get_examples(self, data_dir, split='train'):
        examples = []

        cnt=0
        f=open(os.path.join(data_dir, '%s.tsv'%(split)), encoding='utf-8')
        for i, sent in enumerate(f):
            cnt+=1
            #print("Line %d "%(cnt))

            cls, sent= sent.split('\t')
            cls= cls.lower()
            if cls not in self.get_labels():
                logging.info("Unknown label {}. Skipping line...".format(cls))
                continue
            #print(sent, " ---> " + cls)
            examples.append(
                    InputExample(guid=cnt, text_a=sent, label=cls))
        logging.info("loaded %d classification examples"%(len(examples)))
        return examples
    
    def get_pred_examples(self, file_dir):
        """See base class."""
        return self._create_examples(self._read_file(file_dir),"pred")


    def get_labels(self):
        return ['pos', 'neg'] #'neut']
        
    
    def get_unlabeled_examples(self, data_dir, length = 1000000):
        """See base class."""
        examples = []
        cnt=0
        f=open(os.path.join(data_dir, 'sents.txt'), encoding='utf-8')
        for sent in f:
            cnt+=1

            examples.append(
                    InputExample(guid=cnt, text_a=sent, label=None))
            if cnt>length:
                return examples
        logging.info("loaded %d unlabeled examples"%(len(examples)))
        return examples


    def convert_examples_to_features(self, examples, label_list, max_seq_length, encode_sent_method):

        labels = list(set([e.label for e in examples]))
        logging.info("labels = {}".format(labels))

        label_map= {label_list[i]:i for i in range(len(label_list))}
        max_len_ids = 0
        features=[]
        for e in examples:
            input_ids = encode_sent_method(e.text_a)
            if len(input_ids) > max_len_ids:
                max_len_ids =  len(input_ids)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]


            while len(input_ids) < max_seq_length:
                input_ids.append(1)

            features.append(InputFeatures(input_ids=input_ids,
                label_id= -1 if e.label is None else label_map[e.label]))

        return features, max_len_ids


class SequenceLabelingProcessor:
    """Processor for the CoNLL-2003 data set."""
    def __init__(self, task):
        assert task in ['ner', 'pos']
        if task == 'ner':
             #here I replace the original labels with the extended Wojood labels
            #self.labels = ["O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "U"]
             self.labels = ["O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC","B-NORP","I-NORP",
                           "B-OCC", "I-OCC", "B-FAC", "I-FAC", "B-PRODUCT", "I-PRODUCT",
                           "B-EVENT", "I-EVENT", "B-DATE", "I-DATE", "B-TIME", "I-TIME",
                           "B-LANGUAGE", "I-LANGUAGE", "B-WEBSITE", "I-WEBSITE", "B-LAW", "I-LAW",
                           "B-CARDINAL", "I-CARDINAL", "B-ORDINAL", "I-ORDINAL", "B-PERCENT", "I-PERCENT","B-QUANTITY", "I-QUANTITY",
                           "B-UNIT", "I-UNIT", "B-MONEY", "I-MONEY", "B-CURR", "I-CURR","B-GPE", "I-GPE", "U"]   
        elif task =='pos':
            self.labels = ['TB', 'WB', 'PART', 'V', 'ADJ', 'DET', 'HASH', 'NOUN', 'PUNC',
                           'CONJ', 'PREP', 'PRON', 'EOS', 'CASE', 'EMOT', 'NSUFF', 'NUM',
                                  'URL', 'ADV', 'MENTION', 'FUT_PART', 'ABBREV', 'FOREIGN', 'PROG_PART', 'NEG_PART','U']
            
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt"))
            , "train")

    def get_pred_examples(self, file_dir):
        """See base class."""
        return self._create_examples(self._read_file(file_dir),"pred")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            #self._read_file(os.path.join(data_dir, "test.txt")), "test")
             self._read_file(os.path.join(data_dir, "labeled.txt")), "labeled")
    
    def get_unlabeled_examples(self, data_dir,length=100000):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "Unlabeled.txt")), "unlabeled",length=length)    


    def get_labels(self):
        return self.labels

    def _read_file( self,filename):
        '''
        read file
        '''
        f = open(filename, encoding='utf-8', errors='ignore')
        data = []
        sentence = []
        label = []

        # get all labels in file

        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.' or line.split()[0]=='EOS' or line[0] == '·':
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue

            splits = line.split()
            if len(splits) <= 1:
                continue
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[-1]
            
            if tag not in self.labels:
                continue
            #if tag in ['WB', "TB"]:
            #    tag = "IGNORE"
            if (word == '،' and len(sentence) > 20) or (len(sentence) > 60):
                data.append((sentence, label))
                sentence = []
                label = []
                continue

            sentence.append(word.strip())
            label.append(tag.strip())

        if len(sentence) > 0:
            data.append((sentence, label))
            #print(label)
            sentence = []
            label = []
        return data

    def _create_examples(self, lines, set_type, length=100000):
        examples = []
        c = 0
        for i, (sentence, label) in enumerate(lines):
            c = c + 1
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
            if c > length:
                return examples
        
        logging.info("max sentence length = %d" %(max(len(ex.text_a.split()) for ex in examples)))
        return examples
    
    
    def convert_examples_to_features(self, examples, label_list, max_seq_length, encode_method):
        """Converts a set of examples into XLMR compatible format

        * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
        * Other positions are labeled with 0 ("IGNORE")

        """
        ignored_label = "IGNORE"
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        label_map[ignored_label] = 0  # 0 label is to be ignored
        max_len_ids = 0
        max_str = None
        gt_128 = 0
        features = []
        for (ex_index, example) in enumerate(examples):

            textlist = example.text_a.split(' ')
            labellist = example.label
            labels = []
            token_ids = []
            word_segments = []
            segments_indices_mask = []
            num_tokens = 0
            ## [1,2,3] (word)  ==>  label
            ##
            for i, word in enumerate(textlist):  
                tokens = encode_method(word.strip())  # word token ids
                if len(tokens) ==  0:
                    continue
                token_ids.extend(tokens)  # all sentence token ids
                
                if num_tokens > max_seq_length - 2:
                    continue
                labels.append(labellist[i])
                if num_tokens + len(tokens) > max_seq_length - 2:
                    temp_seg = [j + 1 for j in range(num_tokens, max_seq_length - 1)]
                    #c.append(temp_seg)
                else:
                    temp_seg = [j + 1 for j in range(num_tokens, num_tokens + len(tokens))]
                    #word_segments.append(temp_seg)
                num_tokens = num_tokens + len(tokens)
                
                if len(temp_seg) > 10:
                    temp_seg = temp_seg[:10]
                
                temp_seg_mask = [1 for i in range(len(temp_seg))]
                
                while len(temp_seg) < 10:
                    temp_seg.append(1)  # token padding idx
                    temp_seg_mask.append(0)
                    
                if(sum(temp_seg_mask)) == 0 and i  == 0:
                    print(word.strip())
                    print(len(tokens))
                    print(tokens)
                
                segments_indices_mask.append(temp_seg_mask)
                
                word_segments.append(temp_seg)
                    
            if len(token_ids) > max_len_ids:
                max_len_ids = len(token_ids)
                max_str = example.text_a
            if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
                gt_128 = gt_128+1
                token_ids = token_ids[0:(max_seq_length-2)]
                #labels = labels[0:(max_seq_length-2)]
                #valid = valid[0:(max_seq_length-2)]
                #label_mask = label_mask[0:(max_seq_length-2)]
                

            # adding <s>
            token_ids.insert(0, 0)
            #labels.insert(0, ignored_label)
            #label_mask.insert(0, 0)
            #valid.insert(0, 0)

            # adding </s>
            token_ids.append(2)
            #labels.append(ignored_label)
            #label_mask.append(0)
            #valid.append(0)

   
            #Setence ===> segments (contains subwords)
            #Segment ==> max words = 10
            #sentence ==> max segments = 128
            # Sentence => max words = 128
            
                
            max_segments_length = 0
            segments_mask = [1 if i < len(word_segments) else 0 for i in range(max_seq_length)]
            while len(word_segments) < max_seq_length:
                word_segments.append([0 for i in range(10)])  # token padding idx
                segments_indices_mask.append([0 for i in range(10)])
                labels.append(ignored_label)
                
            input_mask = [1] * len(token_ids)
            
            
            label_ids = []
            for i, _ in enumerate(labels):
                label_ids.append(label_map[labels[i]])
                
            while len(token_ids) < max_seq_length:
                token_ids.append(1)  # token padding idx
                input_mask.append(0)
                #label_ids.append(label_map[ignored_label])  # label ignore idx
                #valid.append(0)
                #label_mask.append(0)


            assert len(token_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(word_segments) == len(label_ids)
            

            features.append(
                InputFeatures(input_ids=token_ids,
                              input_mask=input_mask,
                              labels=label_ids,
                              segments=word_segments,
                             segments_mask = segments_mask,
                             segments_indices_mask=segments_indices_mask))
        #print(f"The number of sentences with more than {max_seq_length} is {gt_128} / {len(examples)}")
        return features, max_len_ids


def create_ner_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.uint8)

    return TensorDataset(
        all_input_ids, all_attention_mask, all_label_ids, all_lmask_ids, all_valid_ids)

def create_clf_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids)

def load_data(data,batchsize=16, num_worker=2, pretraine_path='aubmindlab/bert-base-arabertv2',shuffle=True):
    ENG_train = ReviewDataset(data, pretraine_path)

    ENG_train_loader = DataLoader(dataset=ENG_train, batch_size=batchsize, shuffle=shuffle,
                                   num_workers=num_worker)
    return ENG_train_loader
