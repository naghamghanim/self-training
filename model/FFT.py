import torch

def FFT_CORAL(source, target):
    """
    source: (ns, d)
    target: (nt, d)
    Aligns source and target in the frequency domain
    using covariance of amplitude spectra
    """
    # 1. FFT over feature dimension
    # rfft keeps only non redundant positive frequencies
    source_f = torch.fft.rfft(source, dim=1)
    target_f = torch.fft.rfft(target, dim=1)

    # 2. Use amplitudes only
    source_amp = source_f.abs()      # (ns, df)
    target_amp = target_f.abs()      # (nt, df)

    ns, df_s = source_amp.size(0), source_amp.size(1)
    nt, df_t = target_amp.size(0), target_amp.size(1)
    assert df_s == df_t
    d = df_s

    # 3. Compute covariance in amplitude space

    # center manually as in your CORAL
    # source covariance
    tmp_s = torch.ones((1, ns), device=source_amp.device) @ source_amp
    cs = (
        source_amp.t() @ source_amp
        - (tmp_s.t() @ tmp_s) / ns
    ) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt), device=target_amp.device) @ target_amp
    ct = (
        target_amp.t() @ target_amp
        - (tmp_t.t() @ tmp_t) / nt
    ) / (nt - 1)

    # 4. Frobenius norm between covariance matrices
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss