import numpy as np
from collections import defaultdict

class GTPLSequenceManager:
    """GTPL manager for sequence labeling tasks"""
    def __init__(self, num_labels, base_threshold=0.9, alpha=0.8):
        self.num_labels = num_labels
        self.base_thresh = base_threshold
        self.alpha = alpha  # Start with more weight on source domain
        
        # Initialize thresholds per label
        self.thresholds = torch.ones(num_labels) * base_threshold
        
        # Prototypes: track model confidence per label
        self.source_prototypes = torch.zeros(num_labels)  # From labeled data
        self.target_prototypes = torch.zeros(num_labels)  # From unlabeled data
        self.history_prototypes = torch.zeros(num_labels)  # EMA smoothed
        
        self.ema_decay = 0.9  # For smoothing prototypes
        self.update_count = 0
        
    def update_source_prototypes(self, probs, labels, segments_mask, device='cuda'):
        """Update prototypes from labeled source data"""
        with torch.no_grad():
            batch_size, seq_len, _ = probs.shape
            
            # Flatten for processing
            probs_flat = probs.view(-1, self.num_labels)
            labels_flat = labels.view(-1)
            mask_flat = segments_mask.view(-1)
            
            # Only consider non-padded positions
            valid_mask = mask_flat.bool()
            if valid_mask.sum() == 0:
                return
            
            probs_valid = probs_flat[valid_mask]
            labels_valid = labels_flat[valid_mask]
            
            # Get predictions
            pred_conf, pred_labels = probs_valid.max(dim=1)
            
            # Update prototypes for each label
            for label_idx in range(self.num_labels):
                # Correct predictions in source domain
                correct_mask = (labels_valid == label_idx) & (pred_labels == label_idx)
                if correct_mask.sum() > 0:
                    # Average confidence for correctly predicted tokens
                    self.source_prototypes[label_idx] = pred_conf[correct_mask].mean().item()
    
    def update_target_prototypes(self, probs, segments_mask, device='cuda'):
        """Update prototypes from unlabeled target data"""
        with torch.no_grad():
            batch_size, seq_len, _ = probs.shape
            
            # Flatten for processing
            probs_flat = probs.view(-1, self.num_labels)
            mask_flat = segments_mask.view(-1)
            
            # Only consider non-padded positions
            valid_mask = mask_flat.bool()
            if valid_mask.sum() == 0:
                return
            
            probs_valid = probs_flat[valid_mask]
            
            # Get predictions and confidences
            pred_conf, pred_labels = probs_valid.max(dim=1)
            
            # Update prototypes for each label
            for label_idx in range(self.num_labels):
                # Predictions of this label with confidence above current threshold
                label_mask = (pred_labels == label_idx) & (pred_conf > self.thresholds[label_idx])
                if label_mask.sum() > 0:
                    # Average confidence for high-confidence predictions
                    self.target_prototypes[label_idx] = pred_conf[label_mask].mean().item()
    
    def update_thresholds(self):
        """Update dynamic thresholds based on hybrid prototypes"""
        # Hybrid prototype (weighted by alpha)
        hybrid = self.alpha * self.source_prototypes + (1 - self.alpha) * self.target_prototypes
        
        # EMA smoothing with history
        self.history_prototypes = (
            self.ema_decay * self.history_prototypes + 
            (1 - self.ema_decay) * hybrid
        )
        
        # Normalization (as in GTPL paper)
        max_val = self.history_prototypes.max()
        if max_val < self.base_thresh:
            beta = self.history_prototypes / (max_val + 1e-8)
        else:
            beta = self.history_prototypes
        
        # Update thresholds (clamp to reasonable range)
        new_thresholds = beta * self.base_thresh
        self.thresholds = torch.clamp(new_thresholds, min=0.5, max=0.99)
        
        self.update_count += 1
        # Decay alpha over time (shift weight from source to target)
        if self.update_count % 10 == 0:
            self.alpha = max(0.1, self.alpha * 0.95)  # Gradually decay
    
    def get_token_masks(self, probs, segments_mask, device='cuda'):
        """Get mask for pseudo-label selection using dynamic thresholds"""
        batch_size, seq_len, num_labels = probs.shape
        
        # Get predictions and confidences
        pred_conf, pred_labels = probs.max(dim=2)
        
        # Create mask based on dynamic thresholds
        thresholds_expanded = self.thresholds[pred_labels].to(device)
        mask = (pred_conf > thresholds_expanded) & segments_mask.bool()
        
        return mask, pred_labels