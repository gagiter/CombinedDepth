import torch.nn
import torch
# import segmentation_models_pytorch as smp
import smp


class Vector(torch.nn.Module):
    def __init__(self, name='resnet34', in_channels=3, out_channels=5):
        super(Vector, self).__init__()
        self.net = smp.encoders.get_encoder(name=name, in_channels=in_channels, weights='imagenet')
        last_channels = self.net.out_channels[-1]
        self.conv = torch.nn.Conv2d(last_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.net(x)
        x = self.conv(x[-1])
        x = torch.sigmoid(x)
        x = x.mean(dim=[-2, -1]) - 0.5
        return x


class Matrix(torch.nn.Module):
    def __init__(self, name='resnet34', in_channels=3, out_channels=3):
        super(Matrix, self).__init__()
        self.net = smp.Unet(
            name, encoder_weights='imagenet', in_channels=in_channels,
            classes=out_channels, activation='sigmoid')

    def forward(self, x):
        x = self.net(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder='resnet34', depth_scale=1.0, rotation_scale=1.0, translation_scale=1.0):
        super(Model, self).__init__()
        self.depth_net = Matrix(encoder, in_channels=3, out_channels=1)
        self.camera_net = Vector(encoder, in_channels=3, out_channels=10)
        self.motion_net = Vector(encoder, in_channels=9, out_channels=12)
        self.depth_scale = depth_scale
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale

    def forward(self, data):
        data_out = dict()
        data_out['depth'] = self.depth_net(data['image'])
        camera = self.camera_net(data['image'])

        data_out['camera'] = camera[:, 0:5] * 0.1
        ground_n = camera[:, 5:8]
        ground_n[:, 1:2] = ground_n[:, 1:2] + 0.5
        ground_n = torch.nn.functional.normalize(ground_n)
        ground_d = camera[:, 8:9] + 0.5
        data_out['ground'] = torch.cat([ground_n, ground_d], dim=1)
        data_out['scale'] = (camera[:, 9:10] + 0.5) * 10.0

        image_stack = torch.cat([data['previous'], data['image'], data['next']], dim=1)
        motion = self.motion_net(image_stack)
        if self.rotation_scale > 0.0:
            motion[:, 0:3] *= self.rotation_scale
            motion[:, 6:9] *= self.rotation_scale
        else:
            motion[:, 0:3] = torch.zeros_like(motion[:, 0:3])
            motion[:, 6:9] = torch.zeros_like(motion[:, 6:9])
        if self.translation_scale > 0.0:
            motion[:, 3:6] *= self.translation_scale
            motion[:, 9:12] *= self.translation_scale
        else:
            motion[:, 3:6] = torch.zeros_like(motion[:, 3:6])
            motion[:, 9:12] = torch.zeros_like(motion[:, 9:12])

        data_out['motion'] = motion
        return data_out
