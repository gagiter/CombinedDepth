
import torch.nn


class Criteria(torch.nn.Module):
    def __init__(self):
        super(Criteria, self).__init__()

    def forward(self, data_in, data_out):
        depth_in = data_in['depth']
        mask = data_in['mask']
        depth_out = data_out['depth']
        loss = ((depth_in - depth_out) * mask).abs().sum() / mask.sum()
        return loss

