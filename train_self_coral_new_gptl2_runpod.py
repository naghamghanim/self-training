import argparse
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from seqeval.scheme import IOB2
import numpy as np
import torch
import torch.nn as nn

import pandas as pd

import torch.optim as optim
import modeling

from utils.train_utils import add_xlmr_args
from types import SimpleNamespace
import random
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data_utils import SequenceLabelingProcessor, load_data
import os
from seqeval.metrics import f1_score, classification_report, accuracy_score

import pickle

from utils.train_utils import get_pseudo_labels_threshold,update_gtpl_thresholds,get_pseudo_labels_gtpl
from utils.train_utils import get_pseudo_labels_threshold_dom_sim

from model.Coral import CORAL
import string
import random
import logging

S = 6
ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = S))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_class_thresholds(thresholds, label_names, run, top_k=10, title="GTPL thresholds"):
    """
    thresholds: torch.Tensor [num_labels] or numpy array/list
    label_names: list[str] length num_labels
    """
    if hasattr(thresholds, "detach"):
        thr = thresholds.detach().cpu().numpy()
    else:
        thr = np.array(thresholds, dtype=np.float32)

    print("\n" + "="*60)
    print(f"{title} | run={run}")
    print(f"min={thr.min():.4f}  max={thr.max():.4f}  mean={thr.mean():.4f}  std={thr.std():.4f}")
    print("-"*60)

    # Print all thresholds (sorted by value)
    order = np.argsort(thr)
    for idx in order:
        name = label_names[idx] if idx < len(label_names) else f"label_{idx}"
        print(f"{idx:3d}  {name:25s}  {thr[idx]:.4f}")

    # Optional: highlight extremes
    print("-"*60)
    print("Lowest thresholds:")
    for idx in order[:top_k]:
        name = label_names[idx] if idx < len(label_names) else f"label_{idx}"
        print(f"{idx:3d}  {name:25s}  {thr[idx]:.4f}")

    print("Highest thresholds:")
    for idx in order[-top_k:][::-1]:
        name = label_names[idx] if idx < len(label_names) else f"label_{idx}"
        print(f"{idx:3d}  {name:25s}  {thr[idx]:.4f}")

    print("="*60 + "\n")

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
    logging.info("\n" + classification_report(y_true, y_pred, scheme=IOB2, digits=4))
    print("Classification report done \n")
    print(classification_report(y_true, y_pred, scheme=IOB2, digits=4))
    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", scheme=IOB2),
        "macro_f1": f1_score(y_true, y_pred, average="macro", scheme=IOB2),
        "weights_f1": f1_score(y_true, y_pred, average="weighted", scheme=IOB2),
        "precision": precision_score(y_true, y_pred, scheme=IOB2),
        "recall": recall_score(y_true, y_pred, scheme=IOB2),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if error_analysis:
        return f1, report, y_true, y_pred,SimpleNamespace(**metrics)
    return f1, report,  SimpleNamespace(**metrics)

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
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--unlabeled_data_dir', type=str, default='/workspace/self-training/CONLL-files/')
    parser.add_argument('--indomain', action='store_true', default=False)
    parser.add_argument('--erroranalysis', action='store_true', default=False)
    parser.add_argument('--coral', action='store_true', default=False)
    parser.add_argument('--seg_true', action='store_true', default=False)
    parser.add_argument('--exp_msg', type=str, default='Nothing',help=" for log")
    parser.add_argument("--domain_threshold", default=0.5, type=float,
                    help="Minimum domain similarity threshold")
    parser.add_argument("--domain_sim_file", default="domain_similarity_scores.csv", type=str,
                    help="Path to domain similarity scores CSV file")
    parser.add_argument("--confidence_weight", default=0.5, type=float,
                    help="Weight for confidence in combined scoring")
    parser.add_argument("--domain_weight", default=0.3, type=float,
                    help="Weight for domain similarity in combined scoring")
    
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

    best_val_f1_msa = 0.0
    best_val_f1_da = 0.0    
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


    if args.stop_step == 0:
        config["stop_step"] = 100000
    else:
        config["stop_step"] = args.stop_step

    data_processor = SequenceLabelingProcessor(task=args.task_name)
    label_list = data_processor.get_labels()
    ignore_label_name = "IGNORE"
    all_label_names = label_list + [ignore_label_name]
    num_labels = len(label_list) + 1  # add one for IGNORE label
    if "large" not in args.pretrained_path:
        hidden_size = 768
    else:
        hidden_size = 1024

    classifier = modeling.TokenClassification(pretrained_path=args.pretrained_path,
                                           n_labels=num_labels, hidden_size=hidden_size,
                                           dropout_p=args.dropout, device=device)
    if args.task_name == "ner": 
        # here take the valid PCMA 
        test_datasets =  ['/workspace/self-training/CONLL-files']
    if args.task_name == "pos":
        test_datasets =  ['data/POS-tagging/egy', 'data/POS-tagging/glf','data/POS-tagging/lev','data/POS-tagging/mag','data/POS-tagging/msa']
    val_dataloader = []
    for  val_data in test_datasets:
        val_examples = data_processor.get_dev_examples(val_data)
        val_features, _ = data_processor.convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, classifier.encode_word)
        val_dataloader.append(load_data(val_features,batchsize = args.train_batch_size))
    print("Number of valid sentences ", len(val_examples))
    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    
    triplet = False #not used
    if args.do_train:
        #dir is /workspace/Konooz/Health-domain-cleaned as a source domain
        train_examples = data_processor.get_train_examples(args.data_dir)
        #print(len(train_examples))
        train_features, max_len_ids = data_processor.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, classifier.encode_word)
        print("number of train examples ", len(train_examples))


    if args.self_training:
            # unlabaled_data is the histrory unlabaled /rep/nhamad/Konooz/History-domain-cleaned            
            # Load domain similarity scores (assuming CSV has columns: 'sentence_id' and 'domain_similarity')
            #domain_sim_df = pd.read_csv('/rep/nhamad/AdaSL/Compute-domain-simirlaty/results/domain_similarity_health_history.csv')
            # Create a mapping from sentence index to domain similarity
            '''domain_similarities = {}
            for idx, row in domain_sim_df.iterrows():
                domain_similarities[idx] = row['curriculum_score']
            #read the csv file and select the top N samples based on the curriculum score'''
            self_training_examples = data_processor.get_unlabeled_examples(args.unlabeled_data_dir)
            self_training_features, _ = data_processor.convert_examples_to_features(self_training_examples, label_list, args.max_seq_length, classifier.encode_word)
            unlabeled_dataloader = load_data(self_training_features,batchsize = args.train_batch_size )
            len_train_unlabeled = len(unlabeled_dataloader)
            print("number of unlababled for self-training examples ", len(self_training_examples))
            

    pseudo_labeled_instances = None
    if not args.indomain:
        selected_features_dann = self_training_features[:3000]
    train_dataloader = load_data(train_features,batchsize = args.train_batch_size)


    num_labels = len(label_list) + 1  # you already have this
    base_tau = args.threshold         # e.g. 0.9 or 0.95

    gtpl_thresholds = None            # Τ_{t-1}(c), shape [num_labels]
    gtpl_prototypes = None            # ρ_{t-1}(c), shape [num_labels]


    for run in range(5):
        if args.indomain:
            num_train_optimization_steps = int(
                (len(train_features)) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            num_iterations = (len(train_dataloader)) * args.num_train_epochs
        else:
            print("Selected features for DANN self-training",len(selected_features_dann))
            #print(len(selected_features_dann))
            if len(selected_features_dann) > 0:
                unlab_dataloader = load_data(selected_features_dann,batchsize = args.train_batch_size)

                num_train_optimization_steps = int(max(len(train_features) , len(selected_features_dann)) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
                num_iterations = max(len(train_dataloader) , len(unlab_dataloader)) * args.num_train_epochs
                len_unlab = len(unlab_dataloader)
        
        #psuedo_labels_dataloader ==> unlab_dataloader
               
        test_interval = len(train_dataloader)
        stop_step = 100000
        test_interval = 500
        early_stopping = 10000
        check_es = 0

        base_network = modeling.BertLayer(pretrained_path=args.pretrained_path)
        base_network = base_network.to(device)
        
        #not executed in run 0.
        if run > 0:
            if args.task_name == "ner":
                state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-self-PCMA-econ-{ran}.pt'), 'rb'))
            if args.task_name == "pos":
                state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-GLF-{ran}.pt'), 'rb'))
            base_network.load_state_dict(state_dict)

        hidden_size = base_network.output_num()

        ad_net = modeling.AdversarialNetwork(base_network.output_num(), hidden_size, max_iter=num_iterations)
        ad_net = ad_net.to(device)

        loss_params = config["loss"]
        high = loss_params["trade_off"]

        len_train = len(train_dataloader) # number of training/batchsize
        
        
        classifier = modeling.TokenClassification(pretrained_path=args.pretrained_path,
                                           n_labels=num_labels, hidden_size=hidden_size,
                                           dropout_p=args.dropout, device=device)

        classifier = classifier.to(device)
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)

        model_params = base_network.get_parameters() + classifier.get_parameters() #+ ad_net.get_parameters()

        optimizer = torch.optim.AdamW(model_params,  lr=config["lr"],weight_decay=0.005,)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_iterations)

        #training loop that is moved num_iteration in each run
        for i in range(num_iterations):
            if i == 0:
                print(f"Training started with {device}")
            if i % test_interval == 0 and i > 0:
                check_es = check_es + 500
                base_network.train(False)
                classifier.train(False)
                if args.task_name == "ner":
                    temp_f1_msa, report_msa,metrics = sequence_labeling_test(val_dataloader[0], base_network, classifier,label_list,with_segs = args.seg_true)
                    #temp_f1_da, report_da = sequence_labeling_test(val_dataloader[1], base_network, classifier,label_list,with_segs = args.seg_true)

                    if temp_f1_msa > best_f1_msa:
                        check_es = 0
                        best_step = i
                        best_f1_msa = temp_f1_msa
                        print("\n##########     save the best model self-PCMA-econ.    #############\n")
                        log_str = "iter: {:05d}, Macro F1: {:.5f}".format(best_step, best_f1_msa)
                        print(log_str)
                        print(metrics)
                        metrics_dict = vars(metrics)

                        print("Evaluation metrics")
                        for k, v in metrics_dict.items():
                            print(f"{k:10s}: {float(v):.4f}")
                        torch.save(base_network.state_dict(), open(os.path.join(args.output_dir, f'base_model-self-PCMA-econ-{ran}.pt'), 'wb'))
                        torch.save(classifier.state_dict(), open(os.path.join(args.output_dir, f'model-self-PCMA-econ-{ran}.pt'), 'wb'))
                    log_str = "iter: {:05d}, F1 self-PCMA-econ: {:.5f}, best F1 for self-PCMA-econ: {:.5f}".format(i, temp_f1_msa,best_f1_msa)
                    print(log_str)
                    print(metrics)
                    metrics_dict = vars(metrics)
                    print("Evaluation metrics")
                    for k, v in metrics_dict.items():
                        print(f"{k:10s}: {float(v):.4f}")

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
            

            tinput_ids, tattention_mask, labels_train, t_segments, t_segments_mask, t_segments_indices_mask = next(iter_train)
            tinput_ids, tattention_mask, labels_train, t_segments, t_segments_mask, t_segments_indices_mask = tinput_ids.to(device), tattention_mask.to(device), labels_train.to(device), t_segments.to(device), t_segments_mask, t_segments_indices_mask.to(device)

            bert_features, source_features = base_network(input_ids=tinput_ids, attention_mask=tattention_mask, segments = t_segments, segments_mask = t_segments_mask,segments_indices_mask= t_segments_indices_mask,with_segs = args.seg_true, is_source = True)
            
            train_loss = classifier(bert_features, labels_train, t_segments_mask)

            total_loss = train_loss          
            #print(total_loss)
            #tinput_ids, tattention_mask, labels_train, t_segments, t_segments_mask, t_segments_indices_mask            
            if not args.indomain: # used the self-training
                uinput_ids, uattention_mask, ulabels_train, u_segments, u_segments_mask, u_segments_indices_mask = next(iter_unlab)
                uinput_ids, uattention_mask, ulabels_train, u_segments, u_segments_mask, u_segments_indices_mask = uinput_ids.to(device), uattention_mask.to(device), ulabels_train.to(device), u_segments.to(device), u_segments_mask, u_segments_indices_mask.to(device)


                if uinput_ids.size(0) != tinput_ids.size(0):
                    continue

                unlabeled_features, target_features = base_network(input_ids=uinput_ids, attention_mask=uattention_mask, segments = u_segments, segments_mask = u_segments_mask, segments_indices_mask = u_segments_indices_mask,with_segs = args.seg_true, is_source = False)
                if args.coral:
                    transfer_loss = CORAL(source_features,target_features)
                    total_loss = total_loss + transfer_loss

                high = 1.0
                trade_off = modeling.calc_coeff(run, high=high, max_iter=5)

                if run > 0:
                    
                    pseudo_train_loss = classifier(unlabeled_features, ulabels_train, u_segments_mask)

                    total_loss = total_loss +  pseudo_train_loss
                

            total_loss.backward()
            optimizer.step()
            scheduler.step()

        if args.indomain:
            break

        if args.task_name == "ner":
            state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-self-PCMA-econ-{ran}.pt'), 'rb'))
            base_network.load_state_dict(state_dict)

            state_dict = torch.load(open(os.path.join(args.output_dir, f'model-self-PCMA-econ-{ran}.pt'), 'rb'))
            classifier.load_state_dict(state_dict)
        if args.task_name == "pos":
            state_dict = torch.load(open(os.path.join(args.output_dir, f'base_model-GLF-{ran}.pt'), 'rb'))
            base_network.load_state_dict(state_dict)

            state_dict = torch.load(open(os.path.join(args.output_dir, f'model-GLF-{ran}.pt'), 'rb'))
            classifier.load_state_dict(state_dict)


        print("Updating GTPL thresholds after run ", run) # after training the 4500 iterations
        gtpl_thresholds, gtpl_prototypes = update_gtpl_thresholds(
            base_network=base_network,
            classifier=classifier,
            train_features=train_features,                  # labeled source
            selected_features_dann=selected_features_dann,  # pseudo-labeled target from prev run (None on run=0)
            num_labels=num_labels,
            base_tau=base_tau,
            prev_proto=gtpl_prototypes,
            prev_thresholds=gtpl_thresholds,
            batch_size=args.eval_batch_size,
            device=device,run=run,
            with_segs=args.seg_true
        )
        print("GTPL Thresholds after run ", run)
        print(gtpl_thresholds)  
        print("GTPL Prototypes after run ", run)
        print(gtpl_prototypes)
        print("Selecting pseudo-labeled samples for next run ...")
        #selected_features_dann , _ = get_pseudo_labels_threshold(base_network, classifier, self_training_features, batch_size=args.eval_batch_size, threshold=args.threshold,device = device,with_segs = args.seg_true) # return the selected sentences that are above threshold
        selected_features_dann, _ = get_pseudo_labels_gtpl(
                base_network=base_network,
                classifier=classifier,
                features=self_training_features,
                class_thresholds=gtpl_thresholds,
                batch_size=args.eval_batch_size,
                device=device,
                with_segs=args.seg_true
                    )
        print("Number of selected pseudo-labeled samples: ", len(selected_features_dann))
        log_class_thresholds(gtpl_thresholds, all_label_names, run)
        #selected_features_dann , _ = get_pseudo_labels_threshold_dom_sim(base_network, classifier, self_training_features, batch_size=args.eval_batch_size, threshold=args.threshold, domain_threshold=args.domain_threshold, device = device,with_segs = args.seg_true,domain_similarities=domain_similarities)
    
    #after finished all self-training runs, do a final test on the test set
    if args.task_name == "ner":
        paths_for_best_models = {
        '/workspace/self-training/domain-Ad/variable-K/' : [f'base_model-self-PCMA-econ-{ran}.pt', f'model-self-PCMA-econ-{ran}.pt'],
        
        }
    
    #after finished all self-training runs, do a final test on the test set

    for test_exp in test_datasets:
        print("--------------------------------------")
        '''print(f"Test on {test_exp}")
        state_dict = torch.load(open(os.path.join(args.output_dir, paths_for_best_models[test_exp][0]), 'rb'))
        base_network.load_state_dict(state_dict)

        state_dict = torch.load(open(os.path.join(args.output_dir, paths_for_best_models[test_exp][1]), 'rb'))
        classifier.load_state_dict(state_dict)'''

        base_path = os.path.join(args.output_dir, f'base_model-self-PCMA-econ-{ran}.pt')
        clf_path  = os.path.join(args.output_dir, f'model-self-PCMA-econ-{ran}.pt')

        base_state = torch.load(base_path, map_location=device)
        clf_state  = torch.load(clf_path,  map_location=device)

        base_network.load_state_dict(base_state, strict=True)
        classifier.load_state_dict(clf_state, strict=True)

        base_network.to(device)
        classifier.to(device)


        test_examples = data_processor.get_test_examples(test_exp)
        test_features, _ = data_processor.convert_examples_to_features(
                test_examples, label_list, args.max_seq_length, classifier.encode_word)
        print("number of test (labeled) - target", len(test_features))
        test_dataloader = load_data(test_features,batchsize = args.train_batch_size,shuffle=False)
        
        if args.erroranalysis:
            temp_f1, report, y_truee, y_predd,metrics  = sequence_labeling_test(test_dataloader, base_network, classifier,label_list,with_segs = args.seg_true, error_analysis = True)
            test_texts = [i.text_a for i in test_examples]
            eanalysis = pd.DataFrame(zip(test_texts,y_truee,y_predd),columns = ['Text','True Labels', 'Predicted labels'])
            eanalysis.to_csv(f'error_analysis/{args.exp_msg}_{test_exp.replace("/","_").replace("-","_")}.csv',index=False)
            print(metrics)
            metrics_dict = vars(metrics)

            print("Evaluation metrics")
            for k, v in metrics_dict.items():
                print(f"{k:10s}: {float(v):.4f}")
            print(f"The final F1 on the Test set is {temp_f1}")
            print(f"The final accuracy on the Test set is {report}")
            f = open("results_error_analysis.txt", "a")
            f.write(f"{args.exp_msg} --- {test_exp} --- F1: {temp_f1} --- ACC: {report}\n")
            f.close()
        else:
            temp_f1, report,metrics = sequence_labeling_test(test_dataloader, base_network, classifier,label_list,with_segs = args.seg_true)
            
            print(f"The final F1 on the Test set is {temp_f1}")
            print(f"The final accuracy on the Test set is {report}")
            print(metrics)
            metrics_dict = vars(metrics)

            print("Evaluation metrics")
            for k, v in metrics_dict.items():
                print(f"{k:10s}: {float(v):.4f}")
            f = open("results_arabic_transformers_review.txt", "a")
            f.write(f"{args.exp_msg} --- {test_exp} --- F1: {temp_f1} --- ACC: {report}\n")
            f.close()
        
    for test_exp in test_datasets:
        try:
            os.remove(os.path.join(args.output_dir, paths_for_best_models[test_exp][0]))
            
            os.remove(os.path.join(args.output_dir, paths_for_best_models[test_exp][1]))
        except:
            pass
        #print(report)



