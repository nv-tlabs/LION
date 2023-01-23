import torch
import torch.nn.functional as F

__all__ = ['kl_loss', 'huber_loss']


def kl_loss(x, y):
    x = F.softmax(x.detach(), dim=1)
    y = F.log_softmax(y, dim=1)
    return torch.mean(torch.sum(x * (torch.log(x) - y), dim=1))


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error,
                          torch.full_like(abs_error, fill_value=delta))
    losses = 0.5 * (quadratic**2) + delta * (abs_error - quadratic)
    return torch.mean(losses)
