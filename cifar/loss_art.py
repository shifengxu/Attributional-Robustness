import torch
import torch.nn as nn

ranking_loss = nn.SoftMarginLoss()
cos = nn.CosineSimilarity(dim=1, eps=1e-15)


# given batch size 75, the parameter sizes are:
#   dist_mat_p:torch.Size([150, 150])
#   dist_mat_n:torch.Size([150, 150])
#   labels    :torch.Size([150]) with value [0, 1, 2,,, 73, 74, 0, 1, 2,,, 73, 74]
# Returns distances of attribution map for positive & negative cases
def hard_example_mining(dist_mat_p, dist_mat_n, labels, return_inds=False):
    N = dist_mat_p.size(0)
    # shape [N, N]
    # Tensor.expand(*sizes) → Tensor
    #   >>> x
    #   tensor([1, 2, 3])
    #   >>> x.expand(3, 3)
    #   tensor([[1, 2, 3],
    #           [1, 2, 3],
    #           [1, 2, 3]])
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    # is_pos.shape            : torch.Size([150, 150])
    # dist_mat_p[is_pos].shape: torch.Size([300])
    # dist_mat_p[is_pos] is like:
    # [1.0082, 1.0094, 1.0148, 0.9971, 1.0070, 1.0080, 0.9819, 1.0307, 0.9832,,,]
    # this means, is_pos has 300 "1", and other elements are "0"

    # Tensor.contiguous(memory_format=torch.contiguous_format) → Tensor
    # Returns a contiguous in memory tensor containing the same data as self tensor.
    # ??? seems wrong here.
    # correction:
    # is_pos = torch.eye(N, dtype=torch.int)
    dist_ap, _ = torch.max(dist_mat_p[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat_n[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # now dist_an.shape: [150, 1]

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    # now dist_an.shape: [150]
    # dist_ap is like: [1.0094, 1.0148, 1.0080, 1.0307, 0.9988, 0.9988, 1.0138, 1.0068, 1.0250,,,]

    return dist_ap, dist_an


# Given the parameter sized:
#   x :torch.Size([150, 3072])
#   g1:torch.Size([150, 3072])
#   g2:torch.Size([150, 3072])
# the norm0 will have size [150]
#
# returns the distances for positive & negative cases
# see the page 6 of original paper.
def cosine_dist(x, g1, g2):
    dot_p = x @ g1.t()  # dot product
    norm0 = torch.norm(x, 2, 1) + 1e-8  # p=2 means L2 norm. dim=1.
    norm1 = torch.norm(g1, 2, 1) + 1e-8
    dot_p = torch.div(dot_p, norm0.unsqueeze(1))
    dot_p = torch.div(dot_p, norm1)
    
    dot_n = x @ g2.t()
    norm2 = torch.norm(g2, 2, 1) + 1e-8
    dot_n = torch.div(dot_n, norm0.unsqueeze(1))
    dot_n = torch.div(dot_n, norm2)

    # the cosine similarity dot_p could < 0. so the distance could > 1.
    return 1.0 - dot_p, 1.0 - dot_n


# given batch size 75, the parameter sizes are:
#   x :torch.Size([150, 3072])
#   g1:torch.Size([150, 3072])
#   g2:torch.Size([150, 3072])
#   y :torch.Size([150])
#   a :torch.Size([150]) with value [0, 1, 2,,, 73, 74, 0, 1, 2,,, 73, 74]
# notes: 3072 = 3 * 32 * 32
#
# And other shapes:
#   dist_mat_p:torch.Size([150, 150])
#   dist_mat_n:torch.Size([150, 150])
#   dist_ap   :torch.Size([150])
#   dist_an   :torch.Size([150])
#   loss      :torch.Size([])
def exemplar_loss_fn(x, g1, g2, y, a):
    dist_mat_p, dist_mat_n = cosine_dist(x, g1, g2)
    dist_ap, dist_an = hard_example_mining(dist_mat_p, dist_mat_n, a, return_inds=False)
    # ??? maybe use torch.diagonal(dist_mat_p)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    # in epoch 0, dist_an is like: [1.0260, 1.0034, 0.9874, 0.9810, 0.9860, 0.9726, 0.9986, 0.9859, 0.9934]
    # in epoch 0, dist_ap is like: [1.0157, 1.0141, 0.9863, 1.0228, 1.0096, 1.0041, 1.0019, 0.9709, 0.9969]
    loss = ranking_loss(dist_an - dist_ap, y)
    return loss
