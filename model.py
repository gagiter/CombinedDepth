import torch.nn
import segmentation_models_pytorch as smp


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.depth_net = smp.Unet('resnet34', encoder_weights=None, activation='sigmoid')

    def forward(self, x):
        depth = self.depth_net(x['image'])
        return {'depth': depth}
