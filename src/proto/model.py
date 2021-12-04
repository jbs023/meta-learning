#Reference: https://github.com/tristandeleu/pytorch-meta/tree/master/examples

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def pairwise_cosine_similarity(embeddings1, embeddings2, eps=1e-8):
    sq_norm1 = torch.sum(embeddings1 ** 2, dim=2, keepdim=True)
    sq_norm2 = torch.sum(embeddings2 ** 2, dim=2).unsqueeze(1)
    dot_product = torch.bmm(embeddings1, embeddings2.transpose(1, 2))
    inverse_norm = torch.rsqrt(torch.clamp(sq_norm1 * sq_norm2, min=eps ** 2))
    return dot_product * inverse_norm


def matching_log_probas(embeddings, targets, test_embeddings, num_classes, eps=1e-8):
    batch_size, num_samples, _ = test_embeddings.shape
    similarities = pairwise_cosine_similarity(embeddings, test_embeddings, eps=eps)
    logsumexp = torch.logsumexp(similarities, dim=1, keepdim=True)

    max_similarities, _ = torch.max(similarities, dim=1, keepdim=True)
    exp_similarities = torch.exp(similarities - max_similarities)

    sum_exp = exp_similarities.new_zeros((batch_size, num_classes, num_samples))
    indices = targets.unsqueeze(-1).expand_as(exp_similarities)
    sum_exp.scatter_add_(1, indices, exp_similarities)

    return torch.log(sum_exp) + max_similarities - logsumexp
   

class MatchingNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(MatchingNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )

    def forward(self, train_inputs, train_targets, test_inputs):
        train_embedding = self.encoder(train_inputs.view(-1, *train_inputs.shape[2:]))
        train_embedding = train_embedding.view(*train_inputs.shape[:2], -1)

        test_embedding = self.encoder(test_inputs.view(-1, *test_inputs.shape[2:]))
        test_embedding = test_embedding.view(*test_inputs.shape[:2], -1)

        log_probas = matching_log_probas(train_embedding,
                                        train_targets,
                                        test_embedding,
                                        5)
        return log_probas.exp()