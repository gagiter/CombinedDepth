import torch.nn
import torch
import segmentation_models_pytorch as smp


class Vector(torch.nn.Module):
    def __init__(self, name='resnet34'):
        super(Vector, self).__init__()
        self.net = smp.encoders.get_encoder(name=name)
        last_channels = self.net.out_channels[-1]
        self.conv = torch.nn.Conv2d(last_channels, 5, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.net(x)
        x = self.conv(x[-1])
        x = torch.sigmoid(x)
        x = x.mean(dim=[-2, -1])
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder='resnet34'):
        super(Model, self).__init__()
        self.depth_net = smp.Unet(encoder, encoder_weights=None, activation='sigmoid')
        self.camera_net = Vector(encoder)
        self.motion_net = Vector(encoder)

    def forward(self, data):
        data_out = dict()
        image = data['image']
        data_out['depth'] = self.depth_net(image)
        data_out['camera'] = self.camera_net(image)

        for ref in ['stereo', 'previous', 'next']:
            if ref in data:
                image_ref = data[ref]
                image_stack = torch.cat([image, image_ref], dim=1)
                data_out['motion_' + ref] = self.motion_net(image_stack)

        return data_out
