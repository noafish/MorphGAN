import torch


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    
    return feat_mean, feat_std

def ada_in(feat1, feat2, t):
    
    size = feat1.size()
    mean1, std1 = calc_mean_std(feat1)
    mean2, std2 = calc_mean_std(feat2)
    t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mean3 = (1-t) * mean1 + t * mean2
    std3 = torch.sqrt((1-t) * std1.pow(2) + t * std2.pow(2))

    nfeat1 = (feat1 - mean1.expand(size)) / std1.expand(size)
    nfeat1 = nfeat1 * std3.expand(size) + mean3.expand(size)
    nfeat2 = (feat2 - mean2.expand(size)) / std2.expand(size)
    nfeat2 = nfeat2 * std3.expand(size) + mean3.expand(size)
    return (nfeat1, nfeat2)

