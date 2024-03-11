import torch
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def VICGMLP(mlp, embedding, norm_layer='batch_norm'):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if norm_layer == "batch_norm":
            layers.append(nn.BatchNorm1d(f[i + 1]))
        elif norm_layer == "layer_norm":
            layers.append(nn.LayerNorm(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

def SpectralMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False),
    )

def SimSamMLP(pred_dim=2048, input_dim=512):
    return nn.Sequential(
        nn.Linear(input_dim, pred_dim, bias=False),
        nn.BatchNorm1d(pred_dim),
        nn.ReLU(inplace=True),
        nn.Linear(pred_dim, pred_dim, bias=False),
        nn.BatchNorm1d(pred_dim),
        nn.ReLU(inplace=True),
        nn.Linear(pred_dim, pred_dim, bias=False),
        nn.BatchNorm1d(pred_dim, affine=False),
    )

def PreMLP(dim=2048, pre_dim=512):
    return nn.Sequential(
        nn.Linear(dim, pre_dim, bias=False),
        nn.BatchNorm1d(pre_dim),
        nn.ReLU(inplace=True),
        nn.Linear(pre_dim, dim)
    )


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".
        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss


class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        return loss

class ConLoss(nn.Module):
    def __init__(self, mode = 'simsam', embeding_dim = 512, dim=2048):
        super(ConLoss, self).__init__()
        self.mode = mode # SimSiam, spectral
        print("adopting  ", self.mode)
        if self.mode =='VICReg':
            self.MLP = VICGMLP(mlp='8192-8192-8192', embedding=embeding_dim)
            self.embedding_dim= embeding_dim
        elif self.mode == 'spectral':
            self.MLP = SpectralMLP(dim=embeding_dim, projection_size=1024, hidden_size=4096)
        elif self.mode == 'simsam':
            self.MLP= SimSamMLP(input_dim=embeding_dim)
            self.PreMLP = PreMLP()
            self.criterion= nn.CosineSimilarity(dim=-1)


    def forward(self, feature): #tuple (featurex, featrex_aug)
        if self.mode=='VICReg':
            return self.VICReg(feature)
        elif self.mode=='spectral':
            return self.Spectral(feature, mu=1.0)*0.1
        elif self.mode == 'simsam':
            return self.simsam(feature)
        else:
            return 0


    def simsam(self, feature):
        # print(feature[0].shape, feature[1].shape)
        f1, f2 = self.MLP(feature[0]), self.MLP(feature[1])
        pf1, pf2 = self.PreMLP(f1), self.PreMLP(f2)
        return self.loss_fn_simsiam(self.criterion, pf1, f1.detach(), pf2, f2.detach())

    def loss_fn_simsiam(self, criterion, p1, z1, p2, z2):
        return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    def Spectral(self, feature, mu=1.0):
        z1, z2 = self.MLP(feature[0]), self.MLP(feature[1])

        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                     z1.shape[0] / (z1.shape[0] - 1)
        return (loss_part1 + loss_part2) / mu
        #, {"part1": loss_part1 / mu, "part2": loss_part2 / mu}

    def VICReg(self, feature): #tuple (featurex, featrex_aug)
        feature[0], feature[1] = self.MLP(feature[0]), self.MLP(feature[1])
        num_views = len(feature)
        inv_loss = 0.0
        inv_loss = inv_loss + F.mse_loss(feature[0], feature[1])
        inv_loss =25* inv_loss
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = feature[i]
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + self.off_diagonal(cov_x).pow_(2).sum().div(self.embedding_dim)
            iter_ = iter_ + 1
        var_loss = 25 * var_loss / iter_
        cov_loss = 1 * cov_loss / iter_

        return inv_loss+var_loss+cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

