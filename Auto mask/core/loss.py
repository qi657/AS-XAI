import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.functional import pairwise_distance



class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)  

        mask_pos = mask.masked_fill(eye, 0).float()  
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device)) 
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def calculate_ifd(prototypes):
    """
    Calculate Inter-class Feature Distance given a set of prototypes.

    Args:
        prototypes: A tensor of shape (num_classes, embedding_dim) representing the prototypes
            for each class.

    Returns:
        ifd: A scalar representing the Inter-class Feature Distance.
    """
    num_classes, embedding_dim = prototypes.shape
    pairwise_distances = pairwise_distance(prototypes)
    ifd = (2 / (num_classes * (num_classes - 1))) * pairwise_distances.sum()
    return ifd



def avg_interclass_prototype_distance(prototype_vectors, labels):
    class_prototype_means = {}
    for i in np.unique(labels):
        class_prototype_means[i] = np.mean(prototype_vectors[labels == i], axis=0)

    overall_prototype_mean = np.mean(list(class_prototype_means.values()), axis=0)

    interclass_distances = []
    for i in np.unique(labels):
        distance = np.linalg.norm(class_prototype_means[i] - overall_prototype_mean)
        interclass_distances.append(distance)

    avg_interclass_distance = np.mean(interclass_distances)

    return avg_interclass_distance
