import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from gtpl_ner import GTPL_NER
from pseudo_label_dropout import PseudoLabelDropout


def create_dataloader_from_features(features, batch_size):
    """
    Convert your features to PyTorch DataLoader
    Assuming features structure: (input_ids, attention_mask, token_type_ids, labels)
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    
    # Check if labels exist (for source data)
    if hasattr(features[0], 'label_ids'):
        all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        # For unlabeled data
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    
    return DataLoader(dataset, batch_size=batch_size)

def train_with_gtpl(classifier, args, label_list, 
                   train_features, unlabeled_features):
    """
    Main training function with GTPL
    """
    # ========== 1. Create DataLoaders ==========
    print("Creating data loaders...")
    train_dataloader = create_dataloader_from_features(train_features, args.train_batch_size)
    unlabeled_dataloader = create_dataloader_from_features(unlabeled_features, args.train_batch_size)
    
    # ========== 2. Initialize GTPL and PLD ==========
    print("Initializing GTPL and PLD...")
    gtpl = GTPL_NER(label_list, args)
    pld = PseudoLabelDropout(
        dropout_rate=getattr(args, 'pld_rate', 0.1),
        entity_specific_rates={
            'PER': 0.05,    # Person names are usually reliable
            'ORG': 0.15,    # Organizations vary more by domain
            'LOC': 0.1,     # Locations
            'MISC': 0.2,    # Miscellaneous (most noisy)
            'O': 0.02       # 'O' tag (usually reliable)
        }
    )
    
    # ========== 3. Initialize GTPL from Source ==========
    print("Initializing GTPL prototypes...")
    gtpl.initialize_from_source(classifier, train_dataloader, args.device)
    
    # ========== 4. Training Loop ==========
    print(f"Starting GTPL training for {args.num_train_epochs} epochs...")
    
    for epoch in range(int(args.num_train_epochs)):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        print(f"{'='*60}")
        
        classifier.train()
        total_loss = 0
        total_batches = 0
        
        # Create iterator for unlabeled data
        unlabeled_iter = iter(unlabeled_dataloader)
        
        for batch_idx, batch in enumerate(train_dataloader):
            # ========== Source Batch ==========
            source_input_ids = batch[0].to(args.device)
            source_attention_mask = batch[1].to(args.device)
            source_labels = batch[3].to(args.device)  # Assuming labels at index 3
            
            # Forward on source
            source_outputs = classifier(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                labels=source_labels
            )
            source_loss = source_outputs[0]
            
            # ========== Target Batch ==========
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_dataloader)
                unlabeled_batch = next(unlabeled_iter)
            
            target_input_ids = unlabeled_batch[0].to(args.device)
            target_attention_mask = unlabeled_batch[1].to(args.device)
            
            # Get target predictions (no gradient for pseudo-labeling)
            with torch.no_grad():
                target_outputs = classifier(
                    input_ids=target_input_ids,
                    attention_mask=target_attention_mask
                )
                target_logits = target_outputs[0]
            
            # ========== Generate Pseudo-Labels ==========
            pseudo_labels, confidence_mask = gtpl.generate_pseudo_labels(
                target_logits, target_attention_mask)
            
            # ========== Apply Pseudo Label Dropout ==========
            if getattr(args, 'use_pld', True):
                # Map label indices to entity types for PLD
                label_map = {}
                for entity, indices in gtpl.entity_to_labels.items():
                    for idx in indices:
                        label_map[idx] = entity
                
                pseudo_labels = pld.apply(pseudo_labels, label_map)
            
            # ========== Target Loss ==========
            if torch.any(pseudo_labels != -100):
                # Forward with pseudo-labels
                target_loss_outputs = classifier(
                    input_ids=target_input_ids,
                    attention_mask=target_attention_mask,
                    labels=pseudo_labels
                )
                target_loss = target_loss_outputs[0]
                
                # Progressive weighting: increase target weight over time
                target_weight = min(1.0, (epoch + 1) / 5)  # Warm up over 5 epochs
                weighted_target_loss = target_weight * target_loss
            else:
                weighted_target_loss = 0
            
            # ========== Total Loss ==========
            loss = source_loss + weighted_target_loss
            
            # ========== Backward & Optimize ==========
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            classifier.zero_grad()
            
            # ========== Update GTPL Prototypes ==========
            if batch_idx % getattr(args, 'gtpl_update_freq', 10) == 0:
                with torch.no_grad():
                    # Get fresh predictions
                    fresh_source_logits = classifier(
                        input_ids=source_input_ids,
                        attention_mask=source_attention_mask
                    )[0]
                    
                    fresh_target_logits = classifier(
                        input_ids=target_input_ids,
                        attention_mask=target_attention_mask
                    )[0]
                
                # Update thresholds
                thresholds = gtpl.update_prototypes(
                    fresh_source_logits, source_labels, fresh_target_logits)
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}: Updated GTPL thresholds")
            
            # ========== Logging ==========
            total_loss += loss.item()
            total_batches += 1
            
            if batch_idx % args.logging_steps == 0:
                avg_loss = total_loss / total_batches
                print(f"  Step {batch_idx}: Loss = {avg_loss:.4f}")
                print(f"    Source loss: {source_loss.item():.4f}, Target loss: {weighted_target_loss:.4f}")
                
                # Show pseudo-label coverage
                if torch.any(pseudo_labels != -100):
                    coverage = torch.sum(pseudo_labels != -100).item() / pseudo_labels.numel()
                    print(f"    Pseudo-label coverage: {coverage:.2%}")
        
        # ========== End of Epoch ==========
        avg_epoch_loss = total_loss / total_batches
        print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # ========== Save Checkpoint ==========
        if (epoch + 1) % args.save_steps == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}_gtpl.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gtpl_prototypes': gtpl.prototypes,
                'gtpl_thresholds': gtpl.thresholds,
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # ========== Monitor GTPL Progress ==========
        print("\nCurrent GTPL thresholds (first 10 labels):")
        for i in range(min(10, len(label_list))):
            print(f"  {label_list[i]:15s}: {gtpl.thresholds[i]:.3f}")
    
    print("\n" + "="*60)
    print("GTPL Training Complete!")
    print("="*60)
    
    return classifier, gtpl