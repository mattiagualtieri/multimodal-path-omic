import numpy as np
import torch


class CrossEntropySurvivalLoss:
    def __init__(self, alpha=0.75, eps=1e-7):
        self.alpha = alpha
        self.eps = eps

    def __call__(self, hazards, S, Y, c):
        # batch_size is always 1
        batch_size = len(Y)
        # ground truth
        Y = Y.view(batch_size, 1)
        # censorship status, 0 or 1
        # 0 --> sure patient death
        # 1 --> patient may be alive
        c = c.view(batch_size, 1).float()
        S_padded = torch.cat([torch.ones_like(c), S], 1)
        # This is L_uncensored
        reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=self.eps)) +
                          torch.log(torch.gather(hazards, 1, Y).clamp(min=self.eps)))
        # Bho I don't get the second term
        ce_l = -(c * torch.log(torch.gather(S, 1, Y).clamp(min=self.eps)) +
                 (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=self.eps)))
        loss = (1 - self.alpha) * ce_l + self.alpha * reg
        loss = loss.mean()
        return loss


class NegativeLogLikelihoodSurvivalLoss:
    def __call__(self, hazards, S, Y, c, alpha=0.15, eps=1e-7):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)
        c = c.view(batch_size, 1).float()
        S_padded = torch.cat([torch.ones_like(c), S], 1)
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
            torch.gather(hazards, 1, Y).clamp(min=eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss


class CoxSurvivalLoss:
    def __call__(self, hazards, S, c):
        # https://github.com/traversc/cox-nnet
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        return loss_cox


class SurvivalClassificationTobitLoss:
    def __call__(self, predictions: torch.Tensor, label: torch.Tensor, c: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Compute the Tobit loss for survival analysis with class-based predictions.

        Parameters:
        - predictions: Predicted probabilities for each class (Tensor of shape [4]).
        - label: True class label (scalar tensor).
        - censorship: Binary scalar tensor, where 1 indicates censored data and 0 indicates uncensored data.

        Returns:
        - Loss as a scalar tensor.
        """

        predictions = predictions.reshape(4)
        if c.item() == 0:
            # Uncensored: Use standard cross-entropy loss
            loss = -torch.log(predictions[label] + eps)
        else:
            # Censored: Calculate cumulative probability for survival at least as long as the censored label
            cumulative_prob = torch.sum(predictions[label:])
            loss = -torch.log(cumulative_prob + eps)

        return loss


class CrossEntropySurvivalAttnRegLoss:
    def __init__(self, alpha=0.75, eps=1e-7, lambda_reg=0.01):
        self.alpha = alpha
        self.eps = eps
        self.lambda_reg = lambda_reg
        self.ces = CrossEntropySurvivalLoss(self.alpha, self.eps)

    def __call__(self, hazards, S, Y, c, attention):
        loss = self.ces(hazards, S, Y, c)
        attn_loss = self.lambda_reg * torch.norm(attention, p=2)

        loss = loss + attn_loss
        loss = loss.mean()
        return loss, attn_loss


def test_ces_loss():
    print('Testing CrossEntropySurvivalLoss...')
    loss_function = CrossEntropySurvivalLoss()

    hazards = torch.tensor([0.51, 0.52, 0.49, 0.48]).reshape((1, 4))
    S = torch.tensor([0.5, 0.4, 0.2, 0.1]).reshape((1, 4))
    Y = torch.tensor([0])
    c = torch.tensor([0.0])

    loss = loss_function(hazards, S, Y, c)
    print(f'[1] Loss: {loss.item()}')
    assert loss.item() == 0.6782951951026917

    c = torch.tensor([1.0])

    loss = loss_function(hazards, S, Y, c)
    print(f'[2] Loss: {loss.item()}')
    assert loss.item() == 0.1732867956161499

    print('Test successful')


def test_sct_loss():
    print('Testing SurvivalClassificationTobitLoss...')
    loss_function = SurvivalClassificationTobitLoss()

    Y_pred = torch.tensor([0.1, 0.2, 0.7, 0.1]).reshape((1, 4))
    Y = torch.tensor([2])
    c = torch.tensor([0.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[1] Loss: {loss.item()}\t\t(should be low, because uncensored and guessed correctly)')

    c = torch.tensor([1.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[2] Loss: {loss.item()}\t\t(should be lower than previous, because censored)')

    Y_pred = torch.tensor([0.1, 0.2, 0.7, 0.1]).reshape((1, 4))
    Y = torch.tensor([0])
    c = torch.tensor([0.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[3] Loss: {loss.item()}\t\t(should be high, because uncensored and not guessed correctly)')

    c = torch.tensor([1.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[4] Loss: {loss.item()}\t\t(should be lower than previous, because censored)')

    Y_pred = torch.tensor([0.7, 0.2, 0.1, 0.1]).reshape((1, 4))
    Y = torch.tensor([3])
    c = torch.tensor([0.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[5] Loss: {loss.item()}\t\t(should be high, because uncensored and not guessed correctly)')

    c = torch.tensor([1.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[6] Loss: {loss.item()}\t\t(should be the same, because censored and prediction is completely wrong)')
    
    Y_pred = torch.tensor([0.7, 0.2, 0.1, 0.1]).reshape((1, 4))
    Y = torch.tensor([1])
    c = torch.tensor([1.0])

    loss = loss_function(Y_pred, Y, c)
    print(f'[7] Loss: {loss.item()}\t\t(should be lower than previous, because censored and guess is not far from label)')

    print('Test successful')


def test_cesar_loss():
    print('Testing CrossEntropySurvivalAttnRegLoss...')
    loss_function = CrossEntropySurvivalAttnRegLoss()

    hazards = torch.tensor([0.51, 0.52, 0.49, 0.48]).reshape((1, 4))
    S = torch.tensor([0.5, 0.4, 0.2, 0.1]).reshape((1, 4))
    Y = torch.tensor([0])
    c = torch.tensor([0.0])
    attention = torch.rand(6, 10, 10)

    loss = loss_function(hazards, S, Y, c, attention)
    print(f'[1] Loss: {loss.item()}')

    print('Test successful')
