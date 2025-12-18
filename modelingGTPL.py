import torch
import torch.nn as nn
import torch.nn.functional as F

class GTPL_BERT_NER(nn.Module):
    """
    Wrapper class that integrates BERT NER with GTPL
    """
    def __init__(self, bert_model, num_labels, label_list, args):
        super().__init__()
        
        # Your existing BERT model
        self.bert = bert_model
        
        # Store model configuration
        self.num_labels = num_labels
        self.label_list = label_list
        
        # GTPL parameters
        self.gtpl_enabled = getattr(args, 'use_gtpl', True)
        self.pld_enabled = getattr(args, 'use_pld', True)
        
        # Initialize GTPL and PLD (will be set later)
        self.gtpl = None
        self.pld = None
        
    def set_gtpl(self, gtpl_module):
        """Set GTPL module after initialization"""
        self.gtpl = gtpl_module
    
    def set_pld(self, pld_module):
        """Set PLD module after initialization"""
        self.pld = pld_module
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels=None, is_training=True, use_gtpl_threshold=False):
        """
        Forward pass with optional GTPL integration
        
        Args:
            use_gtpl_threshold: If True, use GTPL thresholds for pseudo-label generation
            is_training: Different behavior for training vs inference
        """
        # Get model outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels if not use_gtpl_threshold else None  # Don't compute loss for pseudo-labels
        )
        
        # Unpack outputs (depends on your BERT model implementation)
        if labels is not None and not use_gtpl_threshold:
            # Supervised forward
            loss, logits = outputs[0], outputs[1]
        else:
            # Unsupervised forward or inference
            logits = outputs[0] if len(outputs) == 1 else outputs[1]
            loss = None
        
        # Apply GTPL for pseudo-label generation during training
        pseudo_labels = None
        if is_training and self.gtpl_enabled and self.gtpl is not None and use_gtpl_threshold:
            # Generate pseudo-labels using GTPL thresholds
            with torch.no_grad():
                pseudo_labels, confidence_mask = self.gtpl.generate_pseudo_labels(
                    logits, attention_mask)
            
            # Apply PLD if enabled
            if self.pld_enabled and self.pld is not None:
                label_map = {}
                if hasattr(self.gtpl, 'entity_to_labels'):
                    for entity, indices in self.gtpl.entity_to_labels.items():
                        for idx in indices:
                            label_map[idx] = entity
                
                pseudo_labels = self.pld.apply(pseudo_labels, label_map)
        
        # Return appropriate values
        if labels is not None and not use_gtpl_threshold:
            return loss, logits
        elif use_gtpl_threshold:
            return pseudo_labels, logits
        else:
            return logits
    
    def get_features(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Extract BERT features (for domain adaptation)
        """
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # Return CLS token embedding or average pooling
        # Last hidden state: [batch_size, seq_len, hidden_dim]
        last_hidden_state = outputs[0]
        
        # Use CLS token for sentence representation
        cls_embeddings = last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        return cls_embeddings