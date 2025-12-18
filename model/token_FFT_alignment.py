import torch

def _masked_token_select(hidden, mask):
    H = hidden.size(-1)
    return hidden[mask].view(-1, H)

def fft_meanvar_align_tokens(source_tokens, target_tokens, eps=1e-6):
    if source_tokens.size(0) < 2 or target_tokens.size(0) < 2:
        return source_tokens.new_tensor(0.0)

    As = torch.log1p(torch.fft.rfft(source_tokens, dim=1).abs())
    At = torch.log1p(torch.fft.rfft(target_tokens, dim=1).abs())

    mu_s, mu_t = As.mean(0), At.mean(0)
    std_s = As.var(0, unbiased=False).add(eps).sqrt()
    std_t = At.var(0, unbiased=False).add(eps).sqrt()

    return (mu_s - mu_t).pow(2).mean() + (std_s - std_t).pow(2).mean()

def fft_entity_alignment_loss(
    source_hidden, target_hidden,
    source_labels, target_pseudo_labels,
    src_valid_mask, tgt_valid_mask,
    target_conf=None, conf_thresh=None,
    o_label_id=0, ignore_label_id=-100
):
    src_valid = src_valid_mask & (source_labels != ignore_label_id)
    tgt_valid = tgt_valid_mask

    src_ent = (source_labels != o_label_id) & src_valid
    tgt_ent = (target_pseudo_labels != o_label_id) & tgt_valid

    if target_conf is not None and conf_thresh is not None:
        tgt_ent = tgt_ent & (target_conf >= conf_thresh)

    src_tokens = _masked_token_select(source_hidden, src_ent)
    tgt_tokens = _masked_token_select(target_hidden, tgt_ent)

    return fft_meanvar_align_tokens(src_tokens, tgt_tokens)
