import torch
import torch.nn as nn
class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error.
    The percentage-based equivalent of L1 Loss.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Calculate the absolute percentage error and take the mean
        loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + self.epsilon)))
        return loss