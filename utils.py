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