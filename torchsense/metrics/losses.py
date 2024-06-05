import torch


def negative_si_snr(x, s, eps=1e-13):
    """
    Args:
        x: Enhanced fo shape [B, T]
        s: Reference of shape [B, T]
    Returns:
        si_snr: [B]
    """

    def l2norm(mat, keep_dim=False):
        return torch.norm(mat, dim=-1, keepdim=keep_dim)

    # Zero-mean x and s
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    s_mean = torch.mean(s, dim=-1, keepdim=True)
    x_zm = x - x_mean
    s_zm = s - s_mean

    # Scaling factor for the target
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keep_dim=True) ** 2 + eps)

    # SI-SNR computation
    si_snr = -torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))

    return si_snr


class ScaleInvariantSignalNoiseRatio(torch.nn.Module):
    def __init__(self):
        super(ScaleInvariantSignalNoiseRatio, self).__init__()

    def forward(self, x, s):
        return negative_si_snr(x, s)