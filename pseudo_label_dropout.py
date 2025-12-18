import torch

class PseudoLabelDropout:
    """
    Randomly drop pseudo-labels to prevent overfitting
    """
    def __init__(self, dropout_rate=0.1, entity_specific_rates=None):
        self.dropout_rate = dropout_rate
        self.entity_specific_rates = entity_specific_rates or {}
    
    def apply(self, pseudo_labels, label_indices=None):
        """
        Randomly set some pseudo-labels to ignore index (-100)
        
        Args:
            pseudo_labels: Tensor of pseudo-labels
            label_indices: Optional dictionary mapping labels to entity types
        
        Returns:
            Dropped pseudo-labels
        """
        if self.dropout_rate == 0:
            return pseudo_labels
        
        dropped = pseudo_labels.clone()
        batch_size, seq_len = pseudo_labels.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                label = pseudo_labels[b, s].item()
                
                # Skip if already ignored
                if label == -100:
                    continue
                
                # Get dropout rate for this label
                if label_indices and label in label_indices:
                    entity_type = label_indices[label]
                    rate = self.entity_specific_rates.get(entity_type, self.dropout_rate)
                else:
                    rate = self.dropout_rate
                
                # Apply dropout
                if torch.rand(1).item() < rate:
                    dropped[b, s] = -100
        
        return dropped