from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report, accuracy_score
import sklearn
import torch
import torch.nn.functional as F
from .data_utils import InputFeatures
import numpy as np
import  pickle

def add_xlmr_args(parser):
     """
     Adds training and validation arguments to the passed parser
     """

     parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--pretrained_path", default=None, type=str, required=True,
                         help="pretrained XLM-Roberta model path")
     parser.add_argument("--task_name",
                         default=None,
                         type=str,
                         required=True,
                         help="The name of the task to train.")
     parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     # Other parameters
     parser.add_argument("--cache_dir",
                         default="",
                         type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3")
     parser.add_argument("--max_seq_length",
                         default=128,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--do_train",
                         action='store_true',
                         help="Whether to run training.")
     parser.add_argument("--do_eval",
                         action='store_true',
                         help="Whether to run eval or not.")
     parser.add_argument("--eval_on",
                         default="dev",
                         help="Whether to run eval on the dev set or test set.")
     parser.add_argument("--do_lower_case",
                         action='store_true',
                         help="Set this flag if you are using an uncased model.")
     parser.add_argument("--train_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--eval_batch_size",
                         default=128,
                         type=int,
                         help="Total batch size for eval.")
     parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--num_train_epochs",
                         default=3,
                         type=int,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--warmup_proportion",
                         default=0.1,
                         type=float,
                         help="Proportion of training to perform linear learning rate warmup for. "
                              "E.g., 0.1 = 10%% of training.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument('--seed',
                         type=int,
                         default=12345,
                         help="random seed for initialization")
     parser.add_argument('--gradient_accumulation_steps',
                         type=int,
                         default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument('--fp16',
                         action='store_true',
                         help="Whether to use 16-bit float precision instead of 32-bit")
     parser.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")
     parser.add_argument('--loss_scale',
                         type=float, default=0,
                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                              "0 (default value): dynamic loss scaling.\n"
                              "Positive power of 2: static loss scaling value.\n")
     parser.add_argument('--dropout', 
                         type=float, default=0.3,
                         help = "training dropout probability")
     
     parser.add_argument('--freeze_model', 
                         action='store_true', default=False,
                         help = "whether to freeze the XLM-R base model and train only the classification heads")
     
     parser.add_argument('--use_crf', 
                         action='store_true', default=False,
                         help = "whether to add a CRF layer on top of the classification head")
     
     parser.add_argument('--no_pbar', 
                         action='store_true', default=False,
                         help = "disable tqdm progress bar")

     
     parser.add_argument('--load_model', 
                         type=str, default=None,
                         help = "saved model to load")



     return parser

def pickle_object(obj,filename):
    outfile = open(filename,'wb')
    pickle.dump(obj,outfile)
    outfile.close()

def compute_class_proto_labeled(base_network, classifier, features, num_labels,
                                batch_size=16, device='cuda', with_segs=True):
    base_network.eval()# set eval mode for inference
    classifier.eval()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segments for f in features])
    all_segments_mask = torch.tensor([f.segments_mask for f in features], dtype=torch.bool)
    all_segments_indices_mask = torch.tensor(
        [f.segments_indices_mask for f in features], dtype=torch.bool)

    sum_conf = torch.zeros(num_labels, device=device) # total confidence per class
    cnt_conf = torch.zeros(num_labels, device=device) # token count per class

    for idx in range(0, all_input_ids.size(0), batch_size):
        input_ids = all_input_ids[idx:idx+batch_size].to(device)
        mask_ids = all_mask_ids[idx:idx+batch_size].to(device)
        label_ids = all_label_ids[idx:idx+batch_size].to(device)
        segments_ids = all_segments[idx:idx+batch_size].to(device)
        segments_mask_ids = all_segments_mask[idx:idx+batch_size].to(device)
        segments_indices_ids = all_segments_indices_mask[idx:idx+batch_size].to(device)

        with torch.no_grad():
            feats, _ = base_network(
                input_ids=input_ids,
                attention_mask=mask_ids,
                segments=segments_ids,
                segments_mask=segments_mask_ids,
                segments_indices_mask=segments_indices_ids,
                with_segs=with_segs,
                is_source=True
            )
            logits = classifier(feats, labels=None) #[B, L, num_labels] [8,128,45]

        #probs, preds = torch.nn.functional.softmax(logits, dim=-1).max(dim=-1)  # B x L
        #probs: confidence in the predicted label     [8,128]     
        #preds: predicted predicted label id (argmax)         [8,128]
        # only real tokens
        probs_all = torch.nn.functional.softmax(logits, dim=-1) 
        '''mask = segments_mask_ids
        correct = (preds == label_ids) & mask #conrrect size [B x L]    [8,128]

        if correct.any():
            probs_flat = probs[correct]         # [N]
            labels_flat = label_ids[correct]    # [N]

            # accumulate sum and count per class
            one_vec = torch.ones_like(probs_flat, device=device)
            sum_conf.scatter_add_(0, labels_flat, probs_flat)
            cnt_conf.scatter_add_(0, labels_flat, one_vec)'''
        gold_ids = label_ids.clamp(min=0, max=num_labels - 1)            # safety
        p_gold = probs_all.gather(dim=-1, index=gold_ids.unsqueeze(-1)).squeeze(-1)  # [B, L]

        mask = segments_mask_ids                                         # [B, L] bool
        valid = mask                                                     # optionally also exclude IGNORE here

        if valid.any():
            p_flat = p_gold[valid]                                       # [N]
            y_flat = gold_ids[valid]                                     # [N]

            one_vec = torch.ones_like(p_flat, device=device)             # [N]
            sum_conf.scatter_add_(0, y_flat, p_flat)
            cnt_conf.scatter_add_(0, y_flat, one_vec)

    proto = torch.zeros_like(sum_conf)
    valid = cnt_conf > 0
    proto[valid] = sum_conf[valid] / cnt_conf[valid]

    return proto  # tensor [num_labels]

def compute_class_proto_unlabeled(base_network, classifier, features, num_labels,
                                  batch_size=16, device='cuda', with_segs=True):
    if features is None or len(features) == 0:
        return torch.zeros(num_labels, device=device)

    base_network.eval()
    classifier.eval()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)  # pseudo labels
    all_segments = torch.tensor([f.segments for f in features])
    all_segments_mask = torch.tensor([f.segments_mask for f in features], dtype=torch.bool)
    all_segments_indices_mask = torch.tensor(
        [f.segments_indices_mask for f in features], dtype=torch.bool)

    sum_conf = torch.zeros(num_labels, device=device)
    cnt_conf = torch.zeros(num_labels, device=device)

    for idx in range(0, all_input_ids.size(0), batch_size):
        input_ids = all_input_ids[idx:idx+batch_size].to(device)
        mask_ids = all_mask_ids[idx:idx+batch_size].to(device)
        plabel_ids = all_label_ids[idx:idx+batch_size].to(device)
        segments_ids = all_segments[idx:idx+batch_size].to(device)
        segments_mask_ids = all_segments_mask[idx:idx+batch_size].to(device)
        segments_indices_ids = all_segments_indices_mask[idx:idx+batch_size].to(device)

        with torch.no_grad():
            feats, _ = base_network(
                input_ids=input_ids,
                attention_mask=mask_ids,
                segments=segments_ids,
                segments_mask=segments_mask_ids,
                segments_indices_mask=segments_indices_ids,
                with_segs=with_segs,
                is_source=False
            )
            logits = classifier(feats, labels=None)

        probs, preds = torch.nn.functional.softmax(logits, dim=-1).max(dim=-1)  # B x L

        # use pseudo labels as “targets” for prototype
        mask = segments_mask_ids
        used = mask  # if you want, you can require preds == plabel_ids here

        if used.any():
            probs_flat = probs[used]
            labels_flat = plabel_ids[used]

            one_vec = torch.ones_like(probs_flat, device=device)
            sum_conf.scatter_add_(0, labels_flat, probs_flat)
            cnt_conf.scatter_add_(0, labels_flat, one_vec)

    proto = torch.zeros_like(sum_conf)
    valid = cnt_conf > 0
    proto[valid] = sum_conf[valid] / cnt_conf[valid]

    return proto
def update_gtpl_thresholds(base_network, classifier,
                           train_features, selected_features_dann,
                           num_labels, base_tau,
                           prev_proto=None, prev_thresholds=None,
                           batch_size=16, device='cuda',run=0, with_segs=True):

    # δ_B and δ_N
    proto_B = compute_class_proto_labeled(
        base_network, classifier, train_features,
        num_labels=num_labels, batch_size=batch_size,
        device=device, with_segs=with_segs
    )

    proto_N = compute_class_proto_unlabeled(
        base_network, classifier, selected_features_dann,
        num_labels=num_labels, batch_size=batch_size,
        device=device, with_segs=with_segs
    )

    # ρ_t(c)
    has_N = proto_N > 0
    #rho = proto_B.clone()
    #rho[has_N] = 0.5 * (proto_B[has_N] + proto_N[has_N])
    alpha0, alpha_min = 1.0, 0.2
    alpha = max(alpha_min, alpha0 - 0.1 * run)  # run = 0,1,2,...
    rho = proto_B.clone()
    rho[has_N] = alpha * proto_B[has_N] + (1.0 - alpha) * proto_N[has_N]
    # σ_t(c) = history smoothed prototype
    if prev_proto is not None:
        sigma = 0.5 * (rho + prev_proto)
    else:
        sigma = rho

    # normalize across classes then scale by base_tau
    max_sigma = sigma.max().item()
    if max_sigma > 0:
        beta = sigma / max_sigma
    else:
        beta = torch.ones_like(sigma)
    
    IGNORE_ID = num_labels - 1
    #thresholds = (beta * base_tau).clamp(0.0, base_tau)
    #thresholds = torch.round(thresholds * 100.0) / 100.0  # 2 decimals
    thresholds = (beta * base_tau).clamp(0.3, base_tau)
    thresholds[IGNORE_ID] = base_tau
    thresholds = torch.round(thresholds * 100.0) / 100.0
    return thresholds, rho

def get_pseudo_labels_gtpl(base_network, classifier, features,
                           class_thresholds, batch_size=16,
                           device='cuda', with_segs=True):
    base_network.eval()
    classifier.eval()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segments for f in features])
    all_segments_mask = torch.tensor([f.segments_mask for f in features], dtype=torch.bool)
    all_segments_indices_mask = torch.tensor(
        [f.segments_indices_mask for f in features], dtype=torch.bool)

    confident_features, rest_features = [], []

    class_thresholds = class_thresholds.to(device)

    for idx in range(0, all_input_ids.size(0), batch_size):
        input_ids = all_input_ids[idx:idx+batch_size].to(device)
        mask_ids = all_mask_ids[idx:idx+batch_size].to(device)
        segments_ids = all_segments[idx:idx+batch_size].to(device)
        segments_mask_ids = all_segments_mask[idx:idx+batch_size].to(device)
        segments_indices_ids = all_segments_indices_mask[idx:idx+batch_size].to(device)

        with torch.no_grad():
            feats, _ = base_network(
                input_ids=input_ids,
                attention_mask=mask_ids,
                segments=segments_ids,
                segments_mask=segments_mask_ids,
                segments_indices_mask=segments_indices_ids,
                with_segs=with_segs,
                is_source=False
            )
            logits = classifier(feats, labels=None)

        probs, preds = torch.nn.functional.softmax(logits, dim=-1).max(dim=-1)  # B x L

        # per-token thresholds T_t(c)
        thr = class_thresholds[preds]  # B x L
        mask = segments_mask_ids

        # token is confident if prob >= T_t(c)
        token_confident = (probs >= thr) & mask

        # sentence is confident if all real tokens are confident
        sent_confident = token_confident.sum(dim=-1) == mask.sum(dim=-1)

        # move to CPU indices
        sent_confident = sent_confident.cpu()

        # slice back the original numpy lists for this batch
        batch_input_ids = all_input_ids[idx:idx+batch_size].cpu().numpy().tolist()
        batch_masks = all_mask_ids[idx:idx+batch_size].cpu().numpy().tolist()
        batch_segments = all_segments[idx:idx+batch_size].cpu().numpy().tolist()
        batch_segments_mask = all_segments_mask[idx:idx+batch_size].cpu().numpy().tolist()
        batch_segments_indices = all_segments_indices_mask[idx:idx+batch_size].cpu().numpy().tolist()
        batch_preds = preds.cpu().numpy().tolist()

        for b in range(len(batch_input_ids)):
            f = InputFeatures(
                input_ids=batch_input_ids[b],
                labels=batch_preds[b],
                input_mask=batch_masks[b],
                segments=batch_segments[b],
                segments_mask=batch_segments_mask[b],
                segments_indices_mask=batch_segments_indices[b],
            )
            if sent_confident[b]:
                confident_features.append(f)
            else:
                rest_features.append(f)

    return confident_features, rest_features

    
def get_pseudo_labels_gtpl_one(base_network, classifier, features, gtpl_manager, 
                          batch_size=16, device='cuda', with_segs=True,num_labels=None):
    """Generate pseudo-labels using GTPL dynamic thresholds"""

    
    base_network.train(False)
    classifier.train(False)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segments for f in features])
    all_segments_indices_mask = torch.tensor([f.segments_indices_mask for f in features], dtype=torch.bool)
    all_segments_mask = torch.tensor([f.segments_mask for f in features], dtype=torch.bool)
    
    confident_features = []
    
    for idx in range(0, all_input_ids.size(0), batch_size):
        # Current batch
        batch_indices = slice(idx, idx + batch_size)
        input_ids = all_input_ids[batch_indices].to(device)
        mask_ids = all_mask_ids[batch_indices].to(device)
        segments_ids = all_segments[batch_indices].to(device)
        segments_indices_ids = all_segments_indices_mask[batch_indices].to(device)
        segments_mask_ids = all_segments_mask[batch_indices].to(device)
        
        with torch.no_grad():
            feature, _ = base_network(
                input_ids=input_ids, 
                attention_mask=mask_ids, 
                segments=segments_ids, 
                segments_mask=segments_mask_ids,
                segments_indices_mask=segments_indices_ids, 
                with_segs=with_segs
            )
            
            logits = classifier(feature, labels=None)
            probs = torch.nn.functional.softmax(logits, dim=2)
        
        # Get token masks using GTPL thresholds
        token_masks, pseudo_labels = gtpl_manager.get_token_masks(probs, segments_mask_ids)
        
        # Move to CPU for processing
        token_masks_cpu = token_masks.cpu()
        pseudo_labels_cpu = pseudo_labels.cpu()
        
        # For each sequence in batch
        for i in range(input_ids.size(0)):
            # Check if sequence has any high-confidence tokens
            if token_masks_cpu[i].sum() > 0:
                # Use pseudo-labels for high-confidence tokens, keep original for others?
                # Or we could use a mix: pseudo-labels where confident, ignore otherwise
                seq_length = segments_mask_ids[i].sum().item()
                
                # Create label sequence: use pseudo-labels where confident, otherwise use IGNORE label
                final_labels = []
                for j in range(seq_length):
                    if token_masks_cpu[i][j]:
                        final_labels.append(pseudo_labels_cpu[i][j].item())
                    else:
                        final_labels.append(num_labels - 1)  # IGNORE label
                
                # Pad to max length
                while len(final_labels) < all_input_ids.size(1):
                    final_labels.append(num_labels - 1)
                
                # Create feature with new labels
                confident_features.append(
                    InputFeatures(
                        input_ids=input_ids[i].cpu().numpy().tolist(),
                        labels=final_labels,
                        input_mask=mask_ids[i].cpu().numpy().tolist(),
                        segments=segments_ids[i].cpu().numpy().tolist(),
                        segments_mask=segments_mask_ids[i].cpu().numpy().tolist(),
                        segments_indices_mask=segments_indices_ids[i].cpu().numpy().tolist()
                    )
                )
    
    return confident_features, []  # Return empty rest_features for simplicity    
    
    
    
def get_pseudo_labels_threshold(base_network, classifier, features, threshold=0.9, batch_size=16, device='cuda',with_segs=True):
    base_network.train(False)
    classifier.train(False)
    y_true = []
    y_pred = []

    filtered_examples = []

    all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.labels for f in features], dtype=torch.long)
    all_segments = torch.tensor(
        [f.segments for f in features])
    all_segments_indices_mask = torch.tensor(
        [f.segments_indices_mask for f in features], dtype=torch.bool)
    all_segments_mask = torch.tensor(
        [f.segments_mask for f in features], dtype=torch.bool)

    predictions = []
    confidences = []

    confident_features, rest_features=[], []
    for idx in range(0, all_input_ids.size(0), batch_size):
        input_ids = all_input_ids[idx:idx+ batch_size]
        mask_ids = all_mask_ids[idx:idx+ batch_size]
        label_ids = all_label_ids[idx:idx+batch_size]
        segments_ids = all_segments[idx:idx+batch_size]
        segments_indices_ids  =  all_segments_indices_mask[idx:idx+batch_size]
        segments_mask_ids = all_segments_mask[idx:idx+batch_size]

        input_ids = input_ids.to(device)
        mask_ids = mask_ids.to(device)
        label_ids = label_ids.to(device)
        segments_ids = segments_ids.to(device)
        segments_indices_ids = segments_indices_ids.to(device)
        segments_mask_ids = segments_mask_ids.to(device)

        with torch.no_grad():
            feature, _ = base_network(input_ids=input_ids, attention_mask=mask_ids, segments = segments_ids, segments_mask = segments_mask_ids,segments_indices_mask= segments_indices_ids, with_segs=with_segs)

            logits = classifier(feature, labels=None)

        prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=2).max(dim=2) # B x L

        prediction_prob[~segments_mask_ids] = 1e7 
        min_confidence  , _= prediction_prob.min(dim=-1)
        
        predictions.append(predicted_labels)
        confidences.append(min_confidence)

    confidences = torch.cat(confidences, dim=0)
    predictions = torch.cat(predictions, dim=0)

    selected_indices = confidences > threshold
    selected_indices = selected_indices.to('cpu')
    selected_input_ids = all_input_ids[selected_indices].cpu().numpy().tolist()
    selected_masks_ids = all_mask_ids[selected_indices].cpu().numpy().tolist()
    selected_label_ids = predictions[selected_indices].cpu().numpy().tolist()
    selected_segments = all_segments[selected_indices].cpu().numpy().tolist()
    selected_segments_indices_mask = all_segments_indices_mask[selected_indices].cpu().numpy().tolist()
    selected_segments_mask = all_segments_mask[selected_indices].cpu().numpy().tolist()
    for ids, masks, labels, segms, segms_ind_msk, segms_msk in zip(selected_input_ids, selected_masks_ids, selected_label_ids, selected_segments, selected_segments_indices_mask, selected_segments_mask):
        confident_features.append(InputFeatures(input_ids=ids, labels=labels, input_mask=masks, segments=segms, segments_mask=segms_msk, segments_indices_mask=segms_ind_msk))


    rest_input_ids = all_input_ids[~selected_indices].cpu().numpy().tolist()
    rest_masks_ids = all_mask_ids[~selected_indices].cpu().numpy().tolist()
    rest_label_ids = predictions[~selected_indices].cpu().numpy().tolist()
    rest_segments = all_segments[~selected_indices].cpu().numpy().tolist()
    rest_segments_indices_mask = all_segments_indices_mask[~selected_indices].cpu().numpy().tolist()
    rest_segments_mask = all_segments_mask[~selected_indices].cpu().numpy().tolist()
                   
    for ids, masks, labels, segms, segms_ind_msk, segms_msk in zip(rest_input_ids, rest_masks_ids, rest_label_ids, rest_segments, rest_segments_indices_mask, rest_segments_mask):
        rest_features.append(InputFeatures(input_ids=ids, labels=labels, input_mask=masks, segments=segms, segments_mask=segms_msk, segments_indices_mask=segms_ind_msk))


    return confident_features, rest_features    
    
##########################################################################################
def get_pseudo_labels_threshold_dom_sim(base_network, classifier, features, threshold=0.9, 
                                domain_threshold=0.7, batch_size=16, device='cuda', 
                                with_segs=True, domain_similarities=None):
    base_network.train(False)
    classifier.train(False)
    y_true = []
    y_pred = []

    filtered_examples = []

    all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.labels for f in features], dtype=torch.long)
    all_segments = torch.tensor(
        [f.segments for f in features])
    all_segments_indices_mask = torch.tensor(
        [f.segments_indices_mask for f in features], dtype=torch.bool)
    all_segments_mask = torch.tensor(
        [f.segments_mask for f in features], dtype=torch.bool)

    predictions = []
    confidences = []

    confident_features, rest_features=[], []

    # Convert domain_similarities to tensor if provided
    #هون في غلط بين حجم ال sim وحجم ال featurs
    if domain_similarities is not None:
        if isinstance(domain_similarities, dict):
            # Create a tensor in the same order as features
            domain_sim_tensor = torch.zeros(len(features))
            for idx, sim in domain_similarities.items():
                if idx < len(features):
                    domain_sim_tensor[idx] = sim
        else:
            domain_sim_tensor = torch.tensor(domain_similarities)
        domain_sim_tensor = domain_sim_tensor.to(device)

    for idx in range(0, all_input_ids.size(0), batch_size):
        input_ids = all_input_ids[idx:idx+ batch_size]
        mask_ids = all_mask_ids[idx:idx+ batch_size]
        label_ids = all_label_ids[idx:idx+batch_size]
        segments_ids = all_segments[idx:idx+batch_size]
        segments_indices_ids  =  all_segments_indices_mask[idx:idx+batch_size]
        segments_mask_ids = all_segments_mask[idx:idx+batch_size]

        input_ids = input_ids.to(device)
        mask_ids = mask_ids.to(device)
        label_ids = label_ids.to(device)
        segments_ids = segments_ids.to(device)
        segments_indices_ids = segments_indices_ids.to(device)
        segments_mask_ids = segments_mask_ids.to(device)

        with torch.no_grad():
            feature, _ = base_network(input_ids=input_ids, attention_mask=mask_ids, segments = segments_ids, segments_mask = segments_mask_ids,segments_indices_mask= segments_indices_ids, with_segs=with_segs)

            logits = classifier(feature, labels=None)

        prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=2).max(dim=2) # B x L

        prediction_prob[~segments_mask_ids] = 1e7 
        min_confidence  , _= prediction_prob.min(dim=-1)
        
        predictions.append(predicted_labels)
        confidences.append(min_confidence)

    confidences = torch.cat(confidences, dim=0)
    predictions = torch.cat(predictions, dim=0)

    # Apply confidence threshold
    confidence_mask = confidences > threshold
    
    # Apply domain similarity threshold if provided
    if domain_similarities is not None:
        domain_mask = domain_sim_tensor > domain_threshold
        selected_indices = confidence_mask & domain_mask
    else:
        selected_indices = confidence_mask
        
    selected_indices = selected_indices.to('cpu')
    selected_input_ids = all_input_ids[selected_indices].cpu().numpy().tolist()
    selected_masks_ids = all_mask_ids[selected_indices].cpu().numpy().tolist()
    selected_label_ids = predictions[selected_indices].cpu().numpy().tolist()
    selected_segments = all_segments[selected_indices].cpu().numpy().tolist()
    selected_segments_indices_mask = all_segments_indices_mask[selected_indices].cpu().numpy().tolist()
    selected_segments_mask = all_segments_mask[selected_indices].cpu().numpy().tolist()
    for ids, masks, labels, segms, segms_ind_msk, segms_msk in zip(selected_input_ids, selected_masks_ids, selected_label_ids, selected_segments, selected_segments_indices_mask, selected_segments_mask):
        confident_features.append(InputFeatures(input_ids=ids, labels=labels, input_mask=masks, segments=segms, segments_mask=segms_msk, segments_indices_mask=segms_ind_msk))


    rest_input_ids = all_input_ids[~selected_indices].cpu().numpy().tolist()
    rest_masks_ids = all_mask_ids[~selected_indices].cpu().numpy().tolist()
    rest_label_ids = predictions[~selected_indices].cpu().numpy().tolist()
    rest_segments = all_segments[~selected_indices].cpu().numpy().tolist()
    rest_segments_indices_mask = all_segments_indices_mask[~selected_indices].cpu().numpy().tolist()
    rest_segments_mask = all_segments_mask[~selected_indices].cpu().numpy().tolist()
                   
    for ids, masks, labels, segms, segms_ind_msk, segms_msk in zip(rest_input_ids, rest_masks_ids, rest_label_ids, rest_segments, rest_segments_indices_mask, rest_segments_mask):
        rest_features.append(InputFeatures(input_ids=ids, labels=labels, input_mask=masks, segments=segms, segments_mask=segms_msk, segments_indices_mask=segms_ind_msk))


    return confident_features, rest_features    


def get_top_confidence_samples_seq_labeling(model, features, batch_size=16,  K=40, device='cuda', balanced=False, n_classes=7):

     """
     Runs model on data, return the set of examples whose prediction confidence is equal of above min_confidence_per_sample
     Args:

        model: the model
        data: set of unlabeled examples 
        min_confidence_per_sample: threshold by which we select examples

    Returns:
        A set of indices of the selected example

     """
     model.eval() # turn of dropout
     y_true = []
     y_pred = []

     filtered_examples = []

     all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
     all_mask_ids = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
     all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
     all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
     all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.uint8)

     predictions = []
     confidences = []

     confident_features, rest_features=[], []
     for idx in range(0, all_input_ids.size(0), batch_size):

        input_ids = all_input_ids[idx:idx+ batch_size]
        mask_ids = all_mask_ids[idx:idx+ batch_size]
        label_ids = all_label_ids[idx:idx+batch_size]
        valid_ids = all_valid_ids[idx:idx+batch_size]
        l_mask = all_lmask_ids[idx:idx+batch_size]

        input_ids = input_ids.to(device)
        mask_ids = mask_ids.to(device)
        valid_ids = valid_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
             logits = model(input_ids, mask_ids, labels=None, labels_mask=l_mask,
                              valid_mask=valid_ids)

        prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=2).max(dim=2) # B x L
        
        prediction_prob[~l_mask.bool()] = 1e7 # so they will be ignored by min
        #prediction_prob[label_ids==0] = 1e7 # ignore WB
        #prediction_prob[label_ids==1] = 1e7 # ignore TB

        min_confidence  , _= prediction_prob.min(dim=-1) # B
        # mean 
        #prediction_prob[~l_mask.bool()] = 0 # so they would be ignored by sum
        #min_confidence = prediction_prob.sum(dim=-1) / l_mask.sum(dim=-1)

        predictions.append(predicted_labels)
        confidences.append(min_confidence)
 
     confidences = torch.cat(confidences, dim=0)
     predictions = torch.cat(predictions, dim=0)
 
     idx_sorted = torch.argsort(confidences, descending=True)
     if K>=1.0:
         K=int(K)
         if balanced:
             pass
         else:
             top_k_idx = idx_sorted[:K]
             rest_idx = idx_sorted[K:]
     else:
        top_k_idx = (confidences >= K)
        rest_idx = (confidences < K)
     
     rest_idx = torch.tensor([i for i in range(len(confidences)) if i not in top_k_idx]).long()
        
     selected_ids = all_input_ids[top_k_idx].cpu().numpy().tolist()
     selected_masks_ids = all_mask_ids[top_k_idx].cpu().numpy().tolist()
     selected_lbls = predictions[top_k_idx].cpu().numpy().tolist()
     selected_masks = all_lmask_ids[top_k_idx].cpu().numpy().tolist()
     selected_valid = all_valid_ids[top_k_idx].cpu().numpy().tolist()

     # add them to examples
     for ids, masks2, lbls, msks, valids in zip(selected_ids, selected_masks_ids, selected_lbls, selected_masks, selected_valid):
         #print(lbls)
         confident_features.append(InputFeatures(input_ids=ids, input_mask=masks2, label_id=lbls, label_mask=msks, valid_ids=valids))
 
     # select those that don't satisfy the confidence
     non_selected_ids = all_input_ids[rest_idx].cpu().numpy().tolist()
     non_selected_masks_ids = all_mask_ids[rest_idx].cpu().numpy().tolist()
     non_selected_lbls = all_label_ids[rest_idx].cpu().numpy().tolist()
     non_selected_masks = all_lmask_ids[rest_idx].cpu().numpy().tolist()
     non_selected_valid = all_valid_ids[rest_idx].cpu().numpy().tolist()
     
     for ids, masks2, lbls, msks, valids in zip(non_selected_ids, non_selected_masks_ids, non_selected_lbls, non_selected_masks, non_selected_valid):
         #print(lbls)
         rest_features.append(InputFeatures(input_ids=ids, input_mask=masks2, label_id=lbls, label_mask=msks, valid_ids=valids))
    
     print(len(rest_features))
     print(len(confident_features))
     print(len(features))
     #assert len(features) == len(rest_features) + len(confident_features) # sanity check
 
     return confident_features, rest_features


def add_self_training_instances(base_network, classifier, data_loader, batch_size=16,  threshold=0.8, device='cuda', balanced=False, n_classes=7):
    base_network.train(False)
    classifier.train(False)

    selected_input_ids = []
    selected_mask_ids = []
    selected_label_ids = []
    selected_valid_ids = []
    selected_lmask_ids = []
    confident_features = []

    
    iter_test = iter(data_loader)
    for i in range(len(data_loader)):
        input_ids, mask_ids, label_ids, l_mask, valid_ids = iter_test.next()
        input_ids, mask_ids, label_ids, l_mask, valid_ids = input_ids.to(device), mask_ids.to(device), label_ids.to(device), l_mask.to(device), valid_ids.to(device)
        with torch.no_grad():
            feature, _ = base_network(input_ids=input_ids, attention_mask=mask_ids, is_train=False)
            logits = classifier(feature, labels=None, labels_mask=l_mask, valid_mask=valid_ids)
            prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=2).max(dim=2) # B x L

        selected_tokens_mask = torch.logical_and(l_mask.bool(), prediction_prob > threshold)

        selected_ids = torch.count_nonzero(selected_tokens_mask, dim=1) !=0

        selected_input_ids.append(input_ids[selected_ids])
        selected_mask_ids.append(mask_ids[selected_ids])
        selected_label_ids.append(predicted_labels[selected_ids])
        selected_valid_ids.append(selected_tokens_mask[selected_ids])
        selected_lmask_ids.append(selected_tokens_mask[selected_ids])

    selected_input_ids = torch.cat(selected_input_ids, dim=0).cpu().numpy().tolist()
    selected_mask_ids = torch.cat(selected_mask_ids, dim=0).cpu().numpy().tolist()
    selected_label_ids = torch.cat(selected_label_ids, dim=0).cpu().numpy().tolist()
    selected_valid_ids = torch.cat(selected_valid_ids, dim=0).cpu().numpy().tolist()
    selected_lmask_ids = torch.cat(selected_lmask_ids, dim=0).cpu().numpy().tolist()

    for ids, masks2, lbls, msks, valids in zip(selected_input_ids, selected_mask_ids, selected_label_ids, selected_lmask_ids, selected_valid_ids):
        confident_features.append(InputFeatures(input_ids=ids, input_mask=masks2, label_id=lbls, label_mask=msks, valid_ids=valids))

    return confident_features


def add_self_training_instances_dann(base_network, classifier, data_loader, batch_size=16,  threshold=0.8, device='cuda', balanced=False, n_classes=7,select_all=None):
    base_network.train(False)
    classifier.train(False)

    selected_input_ids = []
    selected_mask_ids = []
    selected_label_ids = []
    selected_valid_ids = []
    selected_lmask_ids = []
    pseudo_labels = []
    selected_features_samples = []
    no_zeros_couts = []


    
    iter_test = iter(data_loader)
    for i in range(len(data_loader)):
        input_ids, mask_ids, label_ids, l_mask, valid_ids = iter_test.next()
        input_ids, mask_ids, label_ids, l_mask, valid_ids = input_ids.to(device), mask_ids.to(device), label_ids.to(device), l_mask.to(device), valid_ids.to(device)
        with torch.no_grad():
            feature, _ = base_network(input_ids=input_ids, attention_mask=mask_ids, is_train=False)
            logits = classifier(feature, labels=None, labels_mask=l_mask, valid_mask=valid_ids)
            prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=2).max(dim=2) # B x L

        selected_tokens_mask = torch.logical_and(l_mask.bool(), prediction_prob > threshold)

        selected_input_ids.append(input_ids)
        selected_mask_ids.append(mask_ids)
        selected_label_ids.append(predicted_labels)
        selected_valid_ids.append(selected_tokens_mask)
        selected_lmask_ids.append(selected_tokens_mask)
        no_zeros_couts.append(torch.count_nonzero(selected_tokens_mask, dim=1))

    



    if select_all is None:

        conca_zero = torch.cat(no_zeros_couts, dim=0)

        selected_6000_ids = torch.topk(conca_zero, 6000).indices
        pseudo_labels_indices = selected_6000_ids[:3000].cpu().numpy().tolist()
        selected_features_indices = selected_6000_ids[3000:].cpu().numpy().tolist()

        p_selected_input_ids = torch.cat(selected_input_ids, dim=0)[pseudo_labels_indices].cpu().numpy().tolist()
        p_selected_mask_ids = torch.cat(selected_mask_ids, dim=0)[pseudo_labels_indices].cpu().numpy().tolist()
        p_selected_label_ids = torch.cat(selected_label_ids, dim=0)[pseudo_labels_indices].cpu().numpy().tolist()
        p_selected_valid_ids = torch.cat(selected_valid_ids, dim=0)[pseudo_labels_indices].cpu().numpy().tolist()
        p_selected_lmask_ids = torch.cat(selected_lmask_ids, dim=0)[pseudo_labels_indices].cpu().numpy().tolist()

        selected_input_ids = torch.cat(selected_input_ids, dim=0)[selected_features_indices].cpu().numpy().tolist()
        selected_mask_ids = torch.cat(selected_mask_ids, dim=0)[selected_features_indices].cpu().numpy().tolist()
        selected_label_ids = torch.cat(selected_label_ids, dim=0)[selected_features_indices].cpu().numpy().tolist()
        selected_valid_ids = torch.cat(selected_valid_ids, dim=0)[selected_features_indices].cpu().numpy().tolist()
        selected_lmask_ids = torch.cat(selected_lmask_ids, dim=0)[selected_features_indices].cpu().numpy().tolist()

    else:
        p_selected_input_ids = torch.cat(selected_input_ids, dim=0).cpu().numpy().tolist()
        p_selected_mask_ids = torch.cat(selected_mask_ids, dim=0).cpu().numpy().tolist()
        p_selected_label_ids = torch.cat(selected_label_ids, dim=0).cpu().numpy().tolist()
        p_selected_valid_ids = torch.cat(selected_valid_ids, dim=0).cpu().numpy().tolist()
        p_selected_lmask_ids = torch.cat(selected_lmask_ids, dim=0).cpu().numpy().tolist()

        selected_input_ids = torch.cat(selected_input_ids, dim=0).cpu().numpy().tolist()
        selected_mask_ids = torch.cat(selected_mask_ids, dim=0).cpu().numpy().tolist()
        selected_label_ids = torch.cat(selected_label_ids, dim=0).cpu().numpy().tolist()
        selected_valid_ids = torch.cat(selected_valid_ids, dim=0).cpu().numpy().tolist()
        selected_lmask_ids = torch.cat(selected_lmask_ids, dim=0).cpu().numpy().tolist()

    for ids, masks2, lbls, msks, valids in zip(p_selected_input_ids, p_selected_mask_ids, p_selected_label_ids, p_selected_lmask_ids, p_selected_valid_ids):
        pseudo_labels.append(InputFeatures(input_ids=ids, input_mask=masks2, label_id=lbls, label_mask=msks, valid_ids=valids))

    for ids, masks2, lbls, msks, valids in zip(selected_input_ids, selected_mask_ids, selected_label_ids, selected_lmask_ids, selected_valid_ids):
        selected_features_samples.append(InputFeatures(input_ids=ids, input_mask=masks2, label_id=lbls, label_mask=msks, valid_ids=valids))


    return pseudo_labels, selected_features_samples


def get_top_confidence_samples_seq_classification(model, features, batch_size=16,  K=40, device='cuda', balanced=False, n_classes=3):

     """
     Runs model on data, return the set of examples whose prediction confidence is equal of above min_confidence_per_sample
     Args:

        model: the model
        data: set of unlabeled examples 
        min_confidence_per_sample: threshold by which we select examples

    Returns:
        A set of indices of the selected example

     """
     model.eval() # turn of dropout
     y_true = []
     y_pred = []

     filtered_examples = []

     all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    
     all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)


     predictions = []
     confidences = []

     confident_features, rest_features=[], []
     for idx in range(0, all_input_ids.size(0), batch_size):

        input_ids = all_input_ids[idx:idx+ batch_size]
        input_ids = input_ids.to(device)

        with torch.no_grad():
             logits = model(input_ids, labels=None)

        prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=1).max(dim=1) # B 
        
        predictions.append(predicted_labels)
        confidences.append(prediction_prob)
 
     confidences = torch.cat(confidences, dim=0)
     predictions = torch.cat(predictions, dim=0)
 
     idx_sorted = torch.argsort(confidences, descending=True)
     MAX_SAMPLES = 1000

     if K>1.0:
        
         K=int(K)
         if balanced:
             K = K // n_classes
             top_k_idx=[]
             # Taking K examples from each class
             for class_id in range(n_classes):
                 i=0; n_taken=0
                 while i < idx_sorted.size(0) and n_taken < K:
                     idx = idx_sorted[i]
                     if predictions[idx] == class_id:
                         n_taken+=1
                         top_k_idx.append(idx)
                     i += 1
                 if n_taken == K:
                     print("collected all for class %d"%(class_id))

             top_k_idx = torch.LongTensor(top_k_idx)
             rest_idx = torch.tensor([i for i in range(len(confidences)) if i not in top_k_idx]).long()

         else:
             top_k_idx = idx_sorted[:K]
             rest_idx = idx_sorted[K:]
     else:
        top_k_idx = (confidences >= K)
        rest_idx = (confidences < K)
      
 
     selected_ids = all_input_ids[top_k_idx].cpu().numpy().tolist()
     selected_lbls = predictions[top_k_idx].cpu().numpy().tolist()

     unique, counts = np.unique(selected_lbls, return_counts=True)
     frequencies = np.asarray((unique, counts)).T
     print(frequencies)
 
     # add them to examples
     for ids, lbl in zip(selected_ids, selected_lbls):
         confident_features.append(InputFeatures(input_ids=ids, label_id=lbl))
 
     # select those that don't satisfy the confidence
     non_selected_ids = all_input_ids[rest_idx].cpu().numpy().tolist()
     non_selected_lbls = all_label_ids[rest_idx].cpu().numpy().tolist()
     
     for ids, lbl in zip(non_selected_ids, non_selected_lbls):
         #print(lbls)
         rest_features.append(InputFeatures(input_ids=ids, label_id=lbl))

     assert len(features) == len(rest_features) + len(confident_features) # sanity check
 
     return confident_features, rest_features



def evaluate_model_seq_labeling(model, eval_dataset, label_list, batch_size, use_crf, device, pred=False):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     # Run prediction for full data

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list, 1)}
     label_map[0] = "IGNORE"


     for input_ids,mask_ids, label_ids, l_mask, valid_ids in eval_dataset:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)
          mask_ids = mask_ids.to(device)
          valid_ids = valid_ids.to(device)
          l_mask = l_mask.to(device)

          with torch.no_grad():
               logits = model(input_ids, mask_ids, labels=None, labels_mask=None,
                              valid_mask=valid_ids)

          if use_crf:
               predicted_labels = model.decode_logits(logits, mask=l_mask, device=device)
          else :     
               predicted_labels = torch.argmax(logits, dim=2)

          predicted_labels = predicted_labels.detach().cpu().numpy()
          label_ids = label_ids.cpu().numpy()

          for i, cur_label in enumerate(label_ids):
               temp_1 = []
               temp_2 = []

               for j, m in enumerate(cur_label):
                   if valid_ids[i][j] and label_map[m] not in ['WB' , 'TB']: #'PROG_PART', 'NEG_PART']:  # if it's a valid label
                         temp_1.append(label_map[m])
                         temp_2.append(label_map[predicted_labels[i][j]])

               assert len(temp_1) == len(temp_2)
               y_true.append(temp_1)
               y_pred.append(temp_2)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='macro')
     acc = accuracy_score(y_true, y_pred)

     s = "Accuracy = {}".format(acc)
     print(s)
     report +='\n\n'+ s

     if 'NOUN' in label_map.values():
         print("Returning acc")
         f1=acc
    
     if pred:
         return f1, report, y_true, y_pred
     return f1, report

def evaluate_model_seq_classification(model, eval_dataset, label_list, batch_size, device, pred=False):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     from sklearn.metrics import f1_score, classification_report, accuracy_score

     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list)}
     for input_ids, label_ids in eval_dataloader:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None)
               predicted_labels = torch.argmax(logits, dim=1)

          predicted_labels = predicted_labels.detach().cpu().numpy()
          label_ids = label_ids.cpu().numpy()
          y_true.extend(label_ids)
          y_pred.extend(predicted_labels)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='macro')
     acc = accuracy_score(y_true, y_pred)
     
     model.train()

     if pred:
         return f1, report, y_true, y_pred
     return f1, report
