import torch


class CrossEntropySurvivalLoss:
    def __call__(self, hazards, S, Y, c, alpha=0.15):
        return ce_loss(hazards, S, Y, c, alpha)


def ce_loss(hazards, S, Y, c, alpha, eps=1e-7):
    # batch_size is always 1
    batch_size = len(Y)
    # ground truth
    Y = Y.view(batch_size, 1)
    # censorship status, 0 or 1
    # 0 --> certain patient death
    # 1 --> patient may be alive
    c = c.view(batch_size, 1).float()
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # This is L_uncensored
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    # Bho I don't get the second term
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss
