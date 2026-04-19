from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def euclidean_dist_pytorch(x, y):
    dist = torch.cdist(x, y, p=2)
    return dist


def cosine_dist(x, y):
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    x_normalized = x / x_norm
    y_normalized = y / y_norm
    cosine_sim = torch.matmul(x_normalized, y_normalized.transpose(0, 1))
    # cosine_dist = 1 - cosine_sim
    return cosine_sim


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if indice:
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


def _batch_hard2(mat_distance, mat_similarity, indice=False):
    # 对距离矩阵进行处理以找到最难的正样本
    hard_p_matrix = mat_distance + (-9999999.) * (1 - mat_similarity)
    hard_p, hard_p_indice = torch.max(hard_p_matrix, dim=1)

    # 对距离矩阵进行处理以找到最难的负样本
    hard_n_matrix = mat_distance + 9999999. * mat_similarity
    hard_n, hard_n_indice = torch.min(hard_n_matrix, dim=1)

    if indice:
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class TripletLoss(nn.Module):
    '''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

    def __init__(self, margin=0.15, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, emb, label, x):
        emb = emb.squeeze(1)
        label = label.squeeze(1)
        x = x.squeeze(1)
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        # mat_dist = euclidean_dist(emb, emb)
        mat_dist = euclidean_dist_pytorch(emb, emb)
        # mat_dist = cosine_dist(emb, emb)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)

        # mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        # own_mat_sim3 = own_cosine_similarity(label)

        # mat_sim2 = cosine_similarity(label)

        # 初始化相似度矩阵
        mat_sim = F.cosine_similarity(label.unsqueeze(1), label.unsqueeze(0), dim=2)
        input_sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        # dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        dist_ap, dist_an, ap_idx, an_idx = _batch_hard2(mat_dist, mat_sim, indice=True)
        max_ap, max_an, max_ap_idx, max_an_idx = _batch_hard2(mat_dist, input_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)

        # y = torch.ones_like(dist_ap)
        # loss = self.margin_loss(dist_an, dist_ap, y)

        n_label_loss = F.mse_loss(emb, label[an_idx])

        # loss_to_input = F.mse_loss(emb,emb[max_an_idx], reduction='mean')
        # margin_to_input = F.mse_loss(label, emb[max_an_idx], reduction='mean')
        # r = 1
        # input_loss = torch.mean(0.45 - loss_to_input)
        # hard_input_loss = F.mse_loss(emb, x[max_an_idx])
        # print(n_label_loss)
        loss = F.mse_loss(emb, label)+ max(self.margin - n_label_loss, 0)#  + max(self.margin - hard_input_loss, 0))
        # + torch.max(input_loss, 0) + torch.max(self.margin - n_label_loss, 0))
        return loss


class SoftTripletLoss(nn.Module):

    def __init__(self, margin=None, normalize_feature=False):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature

    def forward(self, emb1, emb2, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist_pytorch(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if self.margin is not None:
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        mat_dist_ref = euclidean_dist_pytorch(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss


def _batch_all(mat_distance, mat_similarity, indice=False):
    eq_num = torch.sum(mat_similarity[0]).int()
    ne_num = torch.sum(1 - mat_similarity[0]).int()
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    hard_p = sorted_mat_distance[:, :eq_num]
    hard_p_indice = positive_indices[:, :eq_num]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, :ne_num]
    hard_n_indice = negative_indices[:, :ne_num]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice, eq_num, ne_num
    return hard_p, hard_n, eq_num, ne_num


class NNLoss(nn.Module):

    def __init__(self, margin=None, normalize_feature=False, T=1):
        super(NNLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.T = T

    def forward(self, emb1, emb2, label, epoch=None):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist_pytorch(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx, eq_num, ne_num = _batch_all(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        eq_num -= 1
        if epoch is not None:
            eq_num = max(int(eq_num - (eq_num - 1) * epoch / 40), 1)
            ne_num = max(int(ne_num - (ne_num - 1) * epoch / 40), 1)
        dist_ap = (dist_ap * -1) / self.T
        dist_an = (dist_an * -1) / self.T
        triple_dist = torch.cat((dist_ap[:, :eq_num], dist_an[:, :ne_num]), dim=1)
        triple_dist = F.softmax(triple_dist, dim=1)
        if (self.margin is not None):
            # loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            loss = (- (1 - self.margin) * torch.log(triple_dist[:, :eq_num].sum(dim=1)) - self.margin * torch.log(
                triple_dist[:, :ne_num].sum(dim=1))).mean()
            return loss

        mat_dist_ref = euclidean_dist_pytorch(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx[:, :eq_num]) * (-1) / self.T
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx[:, :ne_num]) * (-1) / self.T
        triple_dist_ref = torch.cat((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * torch.log(triple_dist)).mean(0).sum()
        return loss
