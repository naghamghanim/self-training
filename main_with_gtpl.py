# Add to your existing imports
from gtpl_ner import GTPL_NER
from pseudo_label_dropout import PseudoLabelDropout
from train_with_gtpl import train_with_gtpl

import argparse

import numpy as np
import torch
import torch.nn as nn

import pandas as pd

import torch.optim as optim
import modelingGTPL as modelingGTPL
from modelingGTPL import GTPL_BERT_NER

import modeling

from utils.train_utils import add_xlmr_args

import random
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data_utils import SequenceLabelingProcessor, load_data
import os
from seqeval.metrics import f1_score, classification_report, accuracy_score

import pickle

from utils.train_utils import get_pseudo_labels_threshold

from model.Coral import CORAL
import string
import random
def create_dataloader(features, batch_size, is_labeled=True):
    """
    Create DataLoader from features
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    
    if hasattr(features[0], 'token_type_ids'):
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.zeros_like(all_input_ids)
    
    if is_labeled and hasattr(features[0], 'label_ids'):
        all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
    else:
        dataset = torch.utils.data.TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids
        )
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=is_labeled  # Shuffle only labeled data
    )

def initialize_gtpl_from_source(model, gtpl, dataloader, device, label_list):
    """
    Initialize GTPL prototypes from source domain
    """
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[3].to(device)
            
            # Get model predictions
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs[1] if len(outputs) > 1 else outputs[0]
            
            # Update GTPL with this batch
            gtpl.update_prototypes(logits, labels, None)
            
            # Break after first few batches to save time
            # Or run through entire dataset for better initialization
            break
    
    model.train()
    print("GTPL initialized from source domain")

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create linear schedule with warmup
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

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
    # ... [Your existing setup code] ...

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    ################## Argument Parser ##########
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
    parser.add_argument('--unlabeled_data_dir', type=str, default='/rep/nhamad/Wojood-data/Wojood/CONLL-files/')
    parser.add_argument('--indomain', action='store_true', default=False)
    parser.add_argument('--erroranalysis', action='store_true', default=False)
    parser.add_argument('--coral', action='store_true', default=False)
    parser.add_argument('--seg_true', action='store_true', default=False)
    parser.add_argument('--exp_msg', type=str, default='Nothing',help=" for log")
    
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
    label_list = data_processor.get_labels()
    num_labels = len(label_list) + 1  # add one for IGNORE label
    if "large" not in args.pretrained_path:
        hidden_size = 768
    else:
        hidden_size = 1024

    classifier = modeling.TokenClassification(pretrained_path=args.pretrained_path,
                                           n_labels=num_labels, hidden_size=hidden_size,
                                           dropout_p=args.dropout, device=device)



    # Add GTPL-specific arguments
    args.gtpl_tau = 0.95          # Base threshold
    args.pld_rate = 0.1           # Pseudo-label dropout rate
    args.use_pld = True           # Enable PLD
    args.gtpl_update_freq = 10    # Update every 10 batches
    args.warmup_epochs = 5        # Warm up target loss

    # get the valid of the target domain (Art)
    test_datasets =  ['/rep/nhamad/Wojood-data/Wojood/CONLL-files']
    val_dataloader = []
    for  val_data in test_datasets:
        val_examples = data_processor.get_dev_examples(val_data)
        val_features, _ = data_processor.convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, classifier.encode_word)
        val_dataloader.append(load_data(val_features,batchsize = args.train_batch_size))
    
    print("Number of valid sentences ", len(val_examples))
    # Load your data (as you already do)
    train_examples = data_processor.get_train_examples(args.data_dir)
    train_features, max_len_ids = data_processor.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, classifier.encode_word)
    
    # Load unlabeled data
    self_training_examples = data_processor.get_unlabeled_examples(args.unlabeled_data_dir)
    self_training_features, _ = data_processor.convert_examples_to_features(
        self_training_examples, label_list, args.max_seq_length, classifier.encode_word)
    
    # Train with GTPL
    base_network = modeling.BertLayer(pretrained_path=args.pretrained_path)
    base_network = base_network.to(device)
    hidden_size = base_network.output_num()

    print("Starting training with GTPL for domain adaptation...")
    # ========== 1. Initialize Model (as you already do) ==========
    base_network = modeling.BertLayer(pretrained_path=args.pretrained_path)
    base_network = base_network.to(device)
    
    # ========== 2. Wrap with GTPL Integration ==========
    # Get number of labels from your label_list
    num_labels = len(label_list)  # Should be defined in your code
    
    # Create wrapper
    model = GTPL_BERT_NER(
        bert_model=base_network,
        num_labels=num_labels,
        label_list=label_list,
        args=args
    ).to(device)
    
    # ========== 3. Initialize GTPL and PLD ==========
    gtpl = GTPL_NER(label_list, args)
    pld = PseudoLabelDropout(
        dropout_rate=getattr(args, 'pld_rate', 0.1),
        entity_specific_rates = {
    # ==== PEOPLE & GROUPS (Reliable, common) ====
    'PERS': 0.05,      # Person names - very reliable in Arabic
    'NORP': 0.08,      # Groups/nationalities - fairly reliable
    'OCC': 0.10,       # Occupations - can be ambiguous
    
    # ==== ORGANIZATIONS & PLACES (Medium reliability) ====
    'ORG': 0.12,       # Organizations - variable in news
    'GPE': 0.08,       # Geopolitical entities - usually clear
    'LOC': 0.10,       # Geographical locations - generally reliable
    'FAC': 0.15,       # Facilities/landmarks - can be ambiguous
    
    # ==== TEMPORAL ENTITIES (Very reliable) ====
    'DATE': 0.03,      # Dates - highly reliable patterns
    'TIME': 0.03,      # Time expressions - consistent patterns
    
    # ==== EVENTS & CONCEPTS (Noisy, rare) ====
    'EVENT': 0.20,     # Events - rare and context-dependent
    'LAW': 0.18,       # Laws/legal docs - rare in news
    'PRODUCT': 0.15,   # Products - variable reliability
    
    # ==== NUMERICAL ENTITIES (Very reliable) ====
    'CARDINAL': 0.02,  # Numbers - extremely reliable
    'ORDINAL': 0.02,   # Ordinals - very reliable
    'QUANTITY': 0.05,  # Quantities - mostly reliable
    'UNIT': 0.05,      # Measurement units - reliable
    'MONEY': 0.05,     # Monetary values - reliable patterns
    'PERCENT': 0.03,   # Percentages - very reliable
    
    # ==== OTHER CATEGORIES ====
    'LANGUAGE': 0.12,  # Languages - moderately reliable
    'WEBSITE': 0.10,   # Websites - URL patterns are clear
    'CURRENCY': 0.08,  # Currency names - fairly reliable
    
    # ==== NON-ENTITY (Most common) ====
    'O': 0.01,         # Non-entity tokens - extremely reliable
    })
    
    # Attach to model
    model.set_gtpl(gtpl)
    model.set_pld(pld)
    
    # ========== 4. Load Your Data ==========
    # Your existing data loading code
    train_examples = data_processor.get_train_examples(args.data_dir)
    train_features, max_len_ids = data_processor.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, classifier.encode_word)
    
    # Create dataloader for source data
    train_dataloader = load_data(train_features,batchsize = args.train_batch_size)
    
    # Unlabeled target data
    self_training_examples = data_processor.get_unlabeled_examples(args.unlabeled_data_dir)
    self_training_features, _ = data_processor.convert_examples_to_features(
        self_training_examples, label_list, args.max_seq_length, classifier.encode_word)
    unlabeled_dataloader = load_data(self_training_features,batchsize = args.train_batch_size )
    
    # ========== 5. Initialize GTPL Prototypes ==========
    print("Initializing GTPL prototypes from source domain...")
    initialize_gtpl_from_source(model, gtpl, train_dataloader, device, label_list)
    
    # ========== 6. Setup Optimizer and Scheduler ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Total training steps for scheduler
    total_steps = len(train_dataloader) * args.num_train_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # ========== 7. Training Loop with GTPL ==========
    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        # Set model to training mode
        model.train()
        
        # Iterator for unlabeled data
        unlabeled_iter = iter(unlabeled_dataloader)
        
        for batch_idx, batch in enumerate(train_dataloader):
            # ========== Source Domain (Supervised) ==========
            source_input_ids = batch[0].to(device)
            source_attention_mask = batch[1].to(device)
            source_labels = batch[3].to(device)  # Assuming labels at index 3
            
            # Forward pass with true labels
            source_loss, source_logits = model(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                labels=source_labels,
                is_training=True,
                use_gtpl_threshold=False
            )
            
            # ========== Target Domain (GTPL Pseudo-Labeling) ==========
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_dataloader)
                unlabeled_batch = next(unlabeled_iter)
            
            target_input_ids = unlabeled_batch[0].to(device)
            target_attention_mask = unlabeled_batch[1].to(device)
            
            # Generate pseudo-labels using GTPL thresholds
            pseudo_labels, target_logits = model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                is_training=True,
                use_gtpl_threshold=True  # This triggers GTPL pseudo-label generation
            )
            
            # Compute target loss if we have pseudo-labels
            target_loss = 0
            if torch.any(pseudo_labels != -100):  # Check if any pseudo-labels were generated
                target_loss_outputs = model.bert(
                    input_ids=target_input_ids,
                    attention_mask=target_attention_mask,
                    labels=pseudo_labels
                )
                target_loss = target_loss_outputs[0]
            
            # ========== Combine Losses ==========
            # Progressive weighting: increase target weight over time
            target_weight = min(1.0, (epoch + 1) / 5)  # Ramp up over 5 epochs
            total_loss = source_loss + target_weight * target_loss
            
            # ========== Backward Pass ==========
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # ========== Update Weights ==========
            optimizer.step()
            scheduler.step()
            
            # ========== Update GTPL Prototypes ==========
            if batch_idx % args.gtpl_update_freq == 0:
                # Get fresh predictions (no gradient)
                with torch.no_grad():
                    fresh_source_logits = model.bert(
                        input_ids=source_input_ids,
                        attention_mask=source_attention_mask
                    )[1]  # Get logits (adjust index based on your model)
                    
                    fresh_target_logits = model.bert(
                        input_ids=target_input_ids,
                        attention_mask=target_attention_mask
                    )[1]
                
                # Update GTPL thresholds
                gtpl.update_prototypes(
                    fresh_source_logits, 
                    source_labels, 
                    fresh_target_logits
                )
            
            # ========== Logging ==========
            if batch_idx % args.logging_steps == 0:
                print(f"  Batch {batch_idx}: Loss = {total_loss.item():.4f} "
                      f"(Source: {source_loss.item():.4f}, Target: {target_loss.item():.4f})")
    
    print("\nTraining complete!")
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'gtpl_prototypes': gtpl.prototypes,
        'gtpl_thresholds': gtpl.thresholds,
        'label_list': label_list,
        'args': args
    }, 'bert_ner_gtpl_model.pth')