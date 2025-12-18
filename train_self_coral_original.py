import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd

import torch.optim as optim
import modeling

from utils.train_utils import add_xlmr_args

import random
#from transformers import AdamW, get_linear_schedule_with_warmup

from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data_utils import SequenceLabelingProcessor, load_data
import os
from seqeval.metrics import f1_score, classification_report, accuracy_score

import pickle

from utils.train_utils import get_pseudo_labels_threshold

from model.Coral import CORAL
from model.FFT import FFT_CORAL
from model.token_FFT_alignment import fft_entity_alignment_loss
import string
import random

S = 6
ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = S))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pickle_object(obj,filename):
    outfile = open(filename,'wb')
    pickle.dump(obj,outfile)
    outfile.close()

def sequence_labeling_test(loader, model, classifier, label_list,with_segs, error_analysis = False):
    start_test = True
    dataset = loader

    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "IGNORE"
    model.train(False)
    classifier.train(False)
    
    with torch.no_grad():
        iter_test = iter(dataset)
        y_true = []
        y_pred = []
        for i in range(len(dataset)):
            tinput_ids, tattention_mask, labels_train, segments, segments_mask, segments_indices_mask = next(iter_test)
            
            tinput_ids, tattention_mask, labels_train, segments, segments_mask, segments_indices_mask = tinput_ids.to(device), tattention_mask.to(device), labels_train.to(device), segments.to(device), segments_mask.to(device), segments_indices_mask.to(device)
            
            feature, _ = model(input_ids=tinput_ids, attention_mask=tattention_mask, segments = segments, segments_mask = segments_mask,segments_indices_mask= segments_indices_mask,with_segs = with_segs)
            
            #print("before classiifier")
            
            logits = classifier(feature, labels=None)

            #pickle_object(logits, "logits_test.pickle")
            #print("after classiifier")
            #print(logits)
            #print(logits.shape)
            predicted_labels = torch.argmax(logits, dim=2)
            predicted_labels = predicted_labels.detach().cpu().numpy()
            labels_train = labels_train.cpu().numpy()

            for i, cur_label in enumerate(labels_train):
                 temp_1 = []
                 temp_2 = []

                 for j, m in enumerate(cur_label):
                     if segments_mask[i][j] and label_map[m] not in ['WB' , 'TB']: #'PROG_PART', 'NEG_PART']:  # if it's a valid label
                           temp_1.append(label_map[m])
                           temp_2.append(label_map[predicted_labels[i][j]])

                 assert len(temp_1) == len(temp_2)
                 y_true.append(temp_1)
                 y_pred.append(temp_2)

    report = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    if error_analysis:
        return f1, report, y_true, y_pred
    return f1, report

if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser = add_xlmr_args(parser)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--task', type=str, default='AGJT2MSAC',  help="The dataset or source dataset used")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--restore_dir', type=str, default=None,
                        help="restore directory of our model (in ../snapshot directory)")
    parser.add_argument('--lm_pretrained', type=str, default='gigabert',
                        help=" path of pretrained transformer")
    parser.add_argument('--lr_mult', type=float, default=10, help="dicriminator learning rate multiplier")
    parser.add_argument('--trade_off', type=float, default=1.0,
                        help="trade off between supervised loss and self-training loss")
    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--cos_dist', type=str2bool, default=False, help="the classifier uses cosine similarity.")
    parser.add_argument('--threshold', default=0.9, type=float, help="threshold of pseudo labels")
    parser.add_argument('--label_interval', type=int, default=200, help="interval of two continuous pseudo label phase")
    parser.add_argument('--stop_step', type=int, default=0, help="stop steps")
    parser.add_argument('--final_log', type=str, default=None, help="final_log file")
    parser.add_argument('--weight_type', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='all', help="whether add reg_loss or correct_loss.")
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--test_10crop', type=str2bool, default=True)
    parser.add_argument('--adv_weight', type=float, default=1.0, help="weight of adversarial loss")
    parser.add_argument('--source_detach', default=False, type=str2bool,
                        help="detach source feature from the adversarial learning")
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--self_training', action='store_true', default=False)
    parser.add_argument('--mmd', action='store_true', default=False)
    parser.add_argument('--fft', action='store_true', default=False)
    parser.add_argument('--token_fft', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--unlabeled_data_dir', type=str, default='/rep/nhamad/AdaSL/data/UNLABELED/unlabeled_aoc')
    parser.add_argument('--indomain', action='store_true', default=False)
    parser.add_argument('--erroranalysis', action='store_true', default=False)
    parser.add_argument('--coral', action='store_true', default=False)
    parser.add_argument('--seg_true', action='store_true', default=False)
    parser.add_argument('--exp_msg', type=str, default='Nothing',help=" for log")
    
    do_eval=True
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # train config
    config = {}
    config['args'] = args
    config["gpu"] = args.gpu_id
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["lr"] = args.learning_rate
    config["task"] = args.task
    config["lr_mult"] = args.lr_mult

    config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': args.learning_rate},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.learning_rate, "gamma": 0.001, "power": 0.75}}



    config["loss"] = {"trade_off": args.trade_off}
    config["threshold"] = args.threshold


    if args.stop_step == 0:
        config["stop_step"] = 100000
    else:
        config["stop_step"] = args.stop_step

    data_processor = SequenceLabelingProcessor(task=args.task_name)
    label_list = data_processor.get_labels() # here I replace the original labels with the extended Wojood labels
    num_labels = len(label_list) + 1  # add one for IGNORE label
    if "large" not in args.pretrained_path:
        hidden_size = 768
    else:
        hidden_size = 1024

    classifier = modeling.TokenClassification(pretrained_path=args.pretrained_path,
                                           n_labels=num_labels, hidden_size=hidden_size,
                                           dropout_p=args.dropout, device=device)
    
    data_dir="/rep/nhamad/Konooz/Health-domain-cleaned" # the in-domain data for initial training
    self_training_dir = "/rep/nhamad/Konooz/Art-domain-cleaned" # the out-of-domain data for self-training
    output_dir="/rep/nhamad/AdaSL/Output_konooz_new_experiment" # output directory to save models
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.task_name == "ner":
        #test_datasets =  ['/rep/nhamad/AdaSL/data/NER/Darwish-MSA', '/rep/nhamad/AdaSL/data/NER/Darwish-DA']
        test_datasets =  ['/rep/nhamad/LLM/Wojood1_1_flat'] # only val of Wojood
    if args.task_name == "pos":
        test_datasets =  ['data/POS-tagging/egy', 'data/POS-tagging/glf','data/POS-tagging/lev','data/POS-tagging/mag','data/POS-tagging/msa']
    val_dataloader = []
    for  val_data in test_datasets:
        val_examples = data_processor.get_dev_examples(val_data)
        val_examples = val_examples[:200]
        val_features, _ = data_processor.convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, classifier.encode_word)
        print(f"Validation data directory \"valid.txt\": ")
        print(len(val_examples))
        print("Number of validation examples: ")
        print() 
        #print(f"Number of validation examples in {(os.path.join(val_data, "valid.txt"))}: {len(val_examples)}")
        val_dataloader.append(load_data(val_features,batchsize = args.train_batch_size))

    best_val_f1_msa = 0.0
    best_val_f1_da = 0.0
    

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    #device='cpu'
    
    best_f1_msa = 0.0
    loss_value_msa = 0
    loss_adv_value_msa = 0
    loss_correct_value_msa = 0
    
    best_f1_msa2 = 0.0
    loss_value_msa2 = 0
    loss_adv_value_msa2 = 0
    loss_correct_value_msa2 = 0

    best_f1_da = 0.0
    loss_value_da = 0
    loss_adv_value_da = 0
    loss_correct_value_da = 0

    best_f1_egy = 0.0
    loss_value_egy = 0
    loss_adv_value_egy = 0
    loss_correct_value_egy = 0

    best_f1_glf = 0.0
    loss_value_glf = 0
    loss_adv_value_glf = 0
    loss_correct_value_glf = 0

    best_f1_lev = 0.0
    loss_value_lev = 0
    loss_adv_value_lev = 0
    loss_correct_value_lev = 0
    
    best_f1_mag = 0.0
    loss_value_mag = 0
    loss_adv_value_mag = 0
    loss_correct_value_mag = 0

    
    triplet = False

    if args.do_train:
        #train_examples = data_processor.get_train_examples(args.data_dir)
        train_examples = data_processor.get_train_examples(data_dir)
        #train_examples = train_examples[:2000]
        print("Training data directory: ")
        print(data_dir)
        print("Number of training examples: ")
        print(len(train_examples))
        

        train_features, max_len_ids = data_processor.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, classifier.encode_word)


    if args.self_training:
            #self_training_examples = data_processor.get_unlabeled_examples(args.unlabeled_data_dir)
            self_training_examples = data_processor.get_unlabeled_examples(self_training_dir)
            self_training_examples = self_training_examples[:2000]
            self_training_features, _ = data_processor.convert_examples_to_features(self_training_examples, label_list, args.max_seq_length, classifier.encode_word)
            unlabeled_dataloader = load_data(self_training_features,batchsize = args.train_batch_size )
            len_train_unlabeled = len(unlabeled_dataloader)
            print("Self-training data directory: ")
            print(self_training_dir)
            print("Number of self-training examples: ")
            print(len(self_training_examples))

    pseudo_labeled_instances = None
    if not args.indomain:
        #first we take the first 500 instances for DANN training
        selected_features_dann = self_training_features[:500]
    train_dataloader = load_data(train_features,batchsize = args.train_batch_size)
    
    
    for run in range(5):
        if args.indomain:
            num_train_optimization_steps = int(
                (len(train_features)) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            num_iterations = (len(train_dataloader)) * args.num_train_epochs
        else:
            print("Self-training with DANN")
            print(len(selected_features_dann))
            if len(selected_features_dann) > 0:
                unlab_dataloader = load_data(selected_features_dann,batchsize = args.train_batch_size)
                num_train_optimization_steps = int(max(len(train_features) , len(selected_features_dann)) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
                num_iterations = max(len(train_dataloader) , len(unlab_dataloader)) * args.num_train_epochs
                len_unlab = len(unlab_dataloader)
            
            #psuedo_labels_dataloader ==> unlab_dataloader
                
        test_interval = len(train_dataloader)
        stop_step = 100000
        test_interval = 100 # test every 100 iterations
        early_stopping = 10000
        check_es = 0

        base_network = modeling.BertLayer(pretrained_path=args.pretrained_path)
        base_network = base_network.to(device)
        #ran="8ADVK4"


        

        if run > 0:
            if args.task_name == "ner":
                #state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-MSA-{ran}.pt'), 'rb'))
                state_dict = torch.load(open(os.path.join(output_dir, f'base_model-Konooz-{ran}.pt'), 'rb'))
            if args.task_name == "pos":
                state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-GLF-{ran}.pt'), 'rb'))
            base_network.load_state_dict(state_dict)

        hidden_size = base_network.output_num()

        ad_net = modeling.AdversarialNetwork(base_network.output_num(), hidden_size, max_iter=num_iterations)
        ad_net = ad_net.to(device)

        loss_params = config["loss"]
        high = loss_params["trade_off"]

        len_train = len(train_dataloader)
            

        classifier = modeling.TokenClassification(pretrained_path=args.pretrained_path,
                                            n_labels=num_labels, hidden_size=hidden_size,
                                            dropout_p=args.dropout, device=device)

        classifier = classifier.to(device)
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)

        model_params = base_network.get_parameters() + classifier.get_parameters() #+ ad_net.get_parameters()

            #optimizer = AdamW(model_params, lr=config["lr"], weight_decay=0.005, correct_bias=False)
        optimizer = torch.optim.AdamW(model_params,  lr=config["lr"],weight_decay=0.005,)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_iterations)
            
        pbar = tqdm(range(num_iterations), desc=f"Training ({device})", unit="iter")

        for i in pbar:
            if i == 0:
                pbar.write(f"Training started with {device}")

            if i % test_interval == 0 and i > 0:
                check_es = check_es + 500
                base_network.train(False)
                classifier.train(False)
                if args.task_name == "ner":
                    temp_f1_msa, report_msa = sequence_labeling_test(val_dataloader[0], base_network, classifier,label_list,with_segs = args.seg_true)
                        #temp_f1_da, report_da = sequence_labeling_test(val_dataloader[1], base_network, classifier,label_list,with_segs = args.seg_true)

                    if temp_f1_msa > best_f1_msa:
                        check_es = 0
                        best_step = i
                        best_f1_msa = temp_f1_msa
                        print("\n##########     save the best model MSA.    #############\n")
                        log_str = "iter: {:05d}, Macro F1: {:.5f}".format(best_step, best_f1_msa)
                        print(log_str)
                            #torch.save(base_network.state_dict(), open(os.path.join(args.output_dir, f'base_model-MSA-{ran}.pt'), 'wb'))
                            #torch.save(classifier.state_dict(), open(os.path.join(args.output_dir, f'model-MSA-{ran}.pt'), 'wb'))

                        torch.save(base_network.state_dict(), open(os.path.join(output_dir, f'base_model-Konooz-{ran}.pt'), 'wb'))
                        torch.save(classifier.state_dict(), open(os.path.join(output_dir, f'model-Konooz-{ran}.pt'), 'wb'))
                    log_str = "iter: {:05d}, F1 MSA: {:.5f}, best F1 for MSA: {:.5f}".format(i, temp_f1_msa,best_f1_msa)
                    print(log_str)

                    '''if temp_f1_da > best_f1_da:
                            check_es = 0
                            best_step = i
                            best_f1_da = temp_f1_da
                            print("\n##########     save the best model DA.    #############\n")
                            log_str = "iter: {:05d}, Macro F1: {:.5f}".format(best_step, best_f1_da)
                            print(log_str)
                            torch.save(base_network.state_dict(), open(os.path.join(args.output_dir, f'base_model-DA-{ran}.pt'), 'wb'))
                            torch.save(classifier.state_dict(), open(os.path.join(args.output_dir, f'model-DA-{ran}.pt'), 'wb'))
                        log_str = "iter: {:05d}, F1 DA: {:.5f}, best F1 for DA: {:.5f}".format(i, temp_f1_da,best_f1_da)
                        print(log_str)'''
                    

            if check_es > early_stopping:
                    break
            if i > stop_step:
                    break

                ## train one iter
            base_network.train(True)
            classifier.train(True)
                #ad_net.train(True)
            optimizer.zero_grad()

            if i % len_train == 0:
                iter_train = iter(train_dataloader)

            if not args.indomain:
                if i % len_unlab == 0:
                    iter_unlab = iter(unlab_dataloader)
                

            tinput_ids, tattention_mask, labels_train, t_segments, t_segments_mask, t_segments_indices_mask =next(iter_train)
            tinput_ids, tattention_mask, labels_train, t_segments, t_segments_mask, t_segments_indices_mask = tinput_ids.to(device), tattention_mask.to(device), labels_train.to(device), t_segments.to(device), t_segments_mask, t_segments_indices_mask.to(device)
            src_labels = labels_train
            bert_features, source_features = base_network(input_ids=tinput_ids, attention_mask=tattention_mask, segments = t_segments, segments_mask = t_segments_mask,segments_indices_mask= t_segments_indices_mask,with_segs = args.seg_true, is_source = True)
            o_id = 1

            train_loss = classifier(bert_features, labels_train, t_segments_mask)

            total_loss = train_loss
                
                #print(total_loss)
                
                #tinput_ids, tattention_mask, labels_train, t_segments, t_segments_mask, t_segments_indices_mask
                
            if not args.indomain:
                uinput_ids, uattention_mask, ulabels_train, u_segments, u_segments_mask, u_segments_indices_mask = next(iter_unlab)
                uinput_ids, uattention_mask, ulabels_train, u_segments, u_segments_mask, u_segments_indices_mask = uinput_ids.to(device), uattention_mask.to(device), ulabels_train.to(device), u_segments.to(device), u_segments_mask, u_segments_indices_mask.to(device)


                if uinput_ids.size(0) != tinput_ids.size(0):
                    continue

                unlabeled_features, target_features = base_network(input_ids=uinput_ids, attention_mask=uattention_mask, segments = u_segments, segments_mask = u_segments_mask, segments_indices_mask = u_segments_indices_mask,with_segs = args.seg_true, is_source = False)
                
                if args.coral:
                    #transfer_loss = CORAL(source_features,target_features)
                    #total_loss = total_loss + transfer_loss
                    coral_loss = CORAL(source_features, target_features)
                    #print(f"CORAL loss: {coral_loss.item()}")
                if args.fft:
                    transfer_loss = FFT_CORAL(source_features,target_features)
                    total_loss = total_loss + transfer_loss
                    #fft_loss = FFT_CORAL(source_features, target_features)
                    #print(f"FFT-CORAL loss: {transfer_loss.item()}")

                if args.token_fft:
                    logits_u = classifier(unlabeled_features, labels=None)
                    prob_u, pseudo_u = torch.softmax(logits_u, dim=-1).max(dim=-1)

                    src_valid = t_segments_mask.to(device).bool()
                    tgt_valid = u_segments_mask.to(device).bool()
                    loss_fft_ent = fft_entity_alignment_loss(
                    source_hidden=bert_features,
                    target_hidden=unlabeled_features,
                    source_labels=labels_train,
                    target_pseudo_labels=pseudo_u,
                    src_valid_mask=src_valid,
                    tgt_valid_mask=tgt_valid,
                    target_conf=prob_u,
                    conf_thresh=0.9,
                    o_label_id=1,
                    ignore_label_id=0
                         )
                    
                high = 1.0
                trade_off = modeling.calc_coeff(run, high=high, max_iter=5)

                '''if run > 0:
                    pseudo_train_loss = classifier(unlabeled_features, ulabels_train, u_segments_mask)
                    total_loss = total_loss +  pseudo_train_loss'''
                    
            lambda_fft = 0.5
            total_loss = train_loss + lambda_fft * loss_fft_ent
            total_loss.backward()

            optimizer.step()
            scheduler.step()

        if args.indomain:
            break

        if args.task_name == "ner":
            #state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-MSA-{ran}.pt'), 'rb'))
            #print(os.path.join(args.output_dir, f'base_model-MSA-{ran}.pt'))

            state_dict = torch.load(open(os.path.join(output_dir, f'base_model-Konooz-{ran}.pt'), 'rb'))
            print(os.path.join(output_dir, f'base_model-Konooz-{ran}.pt'))
            print("-----------------------------------------------------------")
            base_network.load_state_dict(state_dict)
                #state_dict = torch.load(open(os.path.join(args.output_dir, f'model-MSA-{ran}.pt'), 'rb'))

            state_dict = torch.load(open(os.path.join(output_dir, f'model-Konooz-{ran}.pt'), 'rb'))
            classifier.load_state_dict(state_dict)

        print("Getting pseudo labels for self-training data...")    
        selected_features_dann , _ = get_pseudo_labels_threshold(base_network, classifier, self_training_features, batch_size=args.eval_batch_size, threshold=args.threshold,device = device,with_segs = args.seg_true)
    
 
    if args.task_name == "ner":
        paths_for_best_models = {
        output_dir : [f'base_model-Konooz-{ran}.pt', f'model-Konooz-{ran}.pt']
        }
        print(paths_for_best_models)

    '''if args.task_name == "ner":
        paths_for_best_models = {
        'data/NER/Darwish-MSA' : [f'base_model-MSA-{ran}.pt', f'model-MSA-{ran}.pt'],
        'data/NER/Darwish-DA' : [f'base_model-DA-{ran}.pt', f'model-DA-{ran}.pt']
        }'''
    if args.task_name == "pos":
        paths_for_best_models = {
        'data/POS-tagging/egy' : [f'base_model-EGY-{ran}.pt', f'model-EGY-{ran}.pt'],
        'data/POS-tagging/glf' : [f'base_model-GLF-{ran}.pt', f'model-GLF-{ran}.pt'],
        'data/POS-tagging/lev' : [f'base_model-LEV-{ran}.pt', f'model-LEV-{ran}.pt'],
        'data/POS-tagging/mag' : [f'base_model-MAG-{ran}.pt', f'model-MAG-{ran}.pt'],
        'data/POS-tagging/msa' : [f'base_model-MSA-{ran}.pt', f'model-MSA-{ran}.pt']
        }

    for test_exp in test_datasets:
        print("--------------------------------------")
        print(f"Test on {test_exp}")
        base_path = os.path.join(output_dir, f'base_model-Konooz-{ran}.pt')
        clf_path  = os.path.join(output_dir, f'model-Konooz-{ran}.pt')

        base_state = torch.load(base_path, map_location=device)
        clf_state  = torch.load(clf_path,  map_location=device)

        base_network.load_state_dict(base_state, strict=True)
        classifier.load_state_dict(clf_state, strict=True)

        base_network.to(device)
        classifier.to(device)

        test_dir="/rep/nhamad/LLM/Wojood1_1_flat"
        test_examples = data_processor.get_test_examples(test_dir)
        test_features, _ = data_processor.convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, classifier.encode_word)

        test_dataloader = load_data(test_features,batchsize = args.train_batch_size,shuffle=False)
        
        if args.erroranalysis:
            temp_f1, report, y_truee, y_predd  = sequence_labeling_test(test_dataloader, base_network, classifier,label_list,with_segs = args.seg_true, error_analysis = True)
            test_texts = [i.text_a for i in test_examples]
            eanalysis = pd.DataFrame(zip(test_texts,y_truee,y_predd),columns = ['Text','True Labels', 'Predicted labels'])
            eanalysis.to_csv(f'error_analysis/{args.exp_msg}_{test_exp.replace("/","_").replace("-","_")}.csv',index=False)
            print(f"The final F1 on the Test set is {temp_f1}")
            print(f"The final accuracy on the Test set is {report}")
            f = open("results_error_analysis.txt", "a")
            f.write(f"{args.exp_msg} --- {test_exp} --- F1: {temp_f1} --- ACC: {report}\n")
            f.close()
        else:
            temp_f1, report = sequence_labeling_test(test_dataloader, base_network, classifier,label_list,with_segs = args.seg_true)
            
            print(f"The final F1 on the Test set is {temp_f1}")
            print(f"The final accuracy on the Test set is {report}")
            f = open("results_arabic_transformers_review.txt", "a")
            f.write(f"{args.exp_msg} --- {test_exp} --- F1: {temp_f1} --- ACC: {report}\n")
            f.close()
        
    for test_exp in test_datasets:
        try:
            #os.remove(os.path.join(args.output_dir, paths_for_best_models[test_exp][0]))
            os.remove(os.path.join(output_dir, paths_for_best_models[test_exp][0]))
            
            #os.remove(os.path.join(args.output_dir, paths_for_best_models[test_exp][1]))
        except:
            pass
        #print(report)



