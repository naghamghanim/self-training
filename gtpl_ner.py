import torch
import torch.nn.functional as F
import numpy as np

class GTPL_NER:
    """
    Growth Threshold for Pseudo Labeling for Arabic NER
    """
    def __init__(self, label_list, args):
        """
        Initialize GTPL with your label list
        
        Args:
            label_list: List of NER labels (e.g., ['O', 'B-PER', 'I-PER', ...])
            args: Training arguments
        """
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.tau = getattr(args, 'gtpl_tau', 0.95)  # Base threshold
        
        # Initialize prototypes (average confidence for each label)
        self.prototypes = torch.ones(self.num_labels) * 0.7
        
        # Initialize thresholds (start high, will adjust)
        self.thresholds = torch.ones(self.num_labels) * self.tau
        
        # Store history for smoothing
        self.history = []
        
        # Map entity types to label indices
        self.build_entity_mapping()
    
    def build_entity_mapping(self):
        """Map entity types (PER, ORG, LOC) to their label indices"""
        self.entity_to_labels = {}
        
        for idx, label in enumerate(self.label_list):
            if label == 'O':
                self.entity_to_labels['O'] = [idx]
            else:
                # Extract entity type from B/I prefix
                if '-' in label:
                    entity_type = label.split('-')[1]
                else:
                    entity_type = label
                
                if entity_type not in self.entity_to_labels:
                    self.entity_to_labels[entity_type] = []
                self.entity_to_labels[entity_type].append(idx)
    
    def initialize_from_source(self, model, dataloader, device):
        """
        Step 1: Initialize prototypes from source domain
        Called BEFORE UDA training begins
        """
        model.eval()
        all_confidences = [[] for _ in range(self.num_labels)]
        
        print("Initializing GTPL prototypes from source domain...")
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch (adjust indices based on your data format)
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[3].to(device)  # Labels at index 3
                
                # Get model predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs[0]  # [batch_size, seq_len, num_labels]
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                confidences, predictions = torch.max(probs, dim=-1)
                
                # Flatten for processing
                batch_size, seq_len = labels.shape
                labels_flat = labels.view(-1)
                predictions_flat = predictions.view(-1)
                confidences_flat = confidences.view(-1)
                
                # Collect confidences for correctly predicted labels
                for label_idx in range(self.num_labels):
                    # Mask for tokens with this label that are predicted correctly
                    correct_mask = (labels_flat == label_idx) & (predictions_flat == label_idx)
                    
                    if correct_mask.any():
                        label_confidences = confidences_flat[correct_mask].cpu().tolist()
                        all_confidences[label_idx].extend(label_confidences)
        
        # Calculate initial prototypes (average confidence for each label)
        for label_idx in range(self.num_labels):
            if all_confidences[label_idx]:
                avg_confidence = np.mean(all_confidences[label_idx])
                self.prototypes[label_idx] = torch.tensor(avg_confidence)
            else:
                # Default confidence for labels without data
                self.prototypes[label_idx] = torch.tensor(0.7)
            
            # Set initial threshold
            self.thresholds[label_idx] = self.prototypes[label_idx] * self.tau
        
        print(f"GTPL initialized with {len(self.label_list)} labels")
        print("Sample prototypes:")
        for i in range(min(5, len(self.label_list))):
            print(f"  {self.label_list[i]}: threshold={self.thresholds[i]:.3f}")
        
        model.train()
    
    def update_prototypes(self, source_logits, source_labels, target_logits):
        """
        Step 2: Update prototypes during training (Eq. 4-10 from paper)
        
        Args:
            source_logits: Predictions on source data
            source_labels: True labels for source data
            target_logits: Predictions on target data
        """
        # Flatten tensors for processing
        source_logits_flat = source_logits.view(-1, self.num_labels)
        source_labels_flat = source_labels.view(-1)
        target_logits_flat = target_logits.view(-1, self.num_labels)
        
        # Get probabilities
        source_probs = F.softmax(source_logits_flat, dim=-1)
        target_probs = F.softmax(target_logits_flat, dim=-1)
        
        source_preds = torch.argmax(source_logits_flat, dim=-1)
        target_preds = torch.argmax(target_logits_flat, dim=-1)
        target_confidences = torch.max(target_probs, dim=-1).values
        
        # Update each label separately
        new_thresholds = torch.zeros_like(self.thresholds)
        
        for label_idx in range(self.num_labels):
            # ========== Source Prototype (Eq. 4) ==========
            # Get correctly predicted source samples for this label
            source_correct = (source_labels_flat == label_idx) & (source_preds == label_idx)
            
            if source_correct.any():
                source_conf = torch.max(source_probs[source_correct], dim=-1).values.mean()
            else:
                source_conf = self.prototypes[label_idx]
            
            # ========== Target Prototype (Eq. 5) ==========
            # Get target predictions above current threshold
            target_mask = (target_preds == label_idx) & (target_confidences > self.thresholds[label_idx])
            
            if target_mask.any():
                target_conf = target_confidences[target_mask].mean()
            else:
                target_conf = torch.tensor(0.0)
            
            # ========== Combined Prototype (Eq. 6) ==========
            if target_conf > 0:
                combined_proto = (source_conf + target_conf) / 2
            else:
                combined_proto = source_conf
            
            # ========== History Smoothing (Eq. 7) ==========
            if len(self.history) > 0:
                old_proto = self.history[-1][label_idx]
                smoothed_proto = (combined_proto + old_proto) / 2
            else:
                smoothed_proto = combined_proto
            
            # Store for next iteration
            self.prototypes[label_idx] = smoothed_proto
            
            # ========== Calculate New Threshold ==========
            # Normalize (Eq. 8)
            max_proto = torch.max(self.prototypes)
            if max_proto < self.tau:
                beta = smoothed_proto / max_proto
            else:
                beta = smoothed_proto
            
            # Threshold calculation (Eq. 9-10)
            new_threshold = beta * self.tau
            
            # Round to 2 decimal places
            new_threshold = torch.round(new_threshold * 100) / 100
            
            new_thresholds[label_idx] = new_threshold
        
        # Update thresholds
        self.thresholds = new_thresholds
        
        # Save current prototypes to history
        self.history.append(self.prototypes.clone())
        
        return self.thresholds
    
    def generate_pseudo_labels(self, logits, attention_mask=None):
        """
        Step 3: Generate pseudo-labels using current thresholds
        
        Returns:
            pseudo_labels: Same shape as input, -100 for ignored tokens
            confidence_mask: Boolean mask showing high-confidence predictions
        """
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        
        # Initialize with ignore index (-100)
        pseudo_labels = torch.full_like(predictions, -100)
        confidence_mask = torch.zeros_like(predictions, dtype=torch.bool)
        
        # Apply per-label threshold
        for label_idx in range(self.num_labels):
            threshold = self.thresholds[label_idx]
            mask = (predictions == label_idx) & (confidences > threshold)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask = mask & (attention_mask == 1)
            
            pseudo_labels[mask] = label_idx
            confidence_mask[mask] = True
        
        return pseudo_labels, confidence_mask