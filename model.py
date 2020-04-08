import torch.nn
import torch
import segmentation_models_pytorch as smp


class Vector(torch.nn.Module):
    def __init__(self, name='resnet34', in_channels=3, out_channels=5):
        super(Vector, self).__init__()
        self.net = smp.encoders.get_encoder(name=name, in_channels=in_channels)
        last_channels = self.net.out_channels[-1]
        self.conv = torch.nn.Conv2d(last_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.net(x)
        x = self.conv(x[-1])
        x = torch.sigmoid(x)
        x = x.mean(dim=[-2, -1]) - 0.5
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder='resnet34', depth_scale=1.0, rotation_scale=1.0, translation_scale=1.0):
        super(Model, self).__init__()
        self.depth_net = smp.Unet(encoder, encoder_weights=None, activation='sigmoid')
        self.camera_net = Vector(encoder, in_channels=3, out_channels=6)
        self.motion_net = Vector(encoder, in_channels=6, out_channels=6)
        self.depth_scale = depth_scale
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale

    def forward(self, data):
        data_out = dict()
        image = data['image']
        # data_out['depth'] = -40.0 * torch.log(data_out['depth'])  # z = exp(-0.025 * d)
        # data_out['depth'] = self.depth_net(image) * 100.0  # z = d * 100.0
        data_out['depth'] = self.depth_net(image)  # z = 1.0 / d
        # if self.depth_scale > 0.0:
        #     data_out['depth'] *= self.depth_scale
        # else:
        #     data_out['depth'] = torch.ones_like(data_out['depth'])
        data_out['camera'] = self.camera_net(image) * 0.01
        data_out['camera'] = torch.zeros_like(data_out['camera'])

        for ref in ['stereo', 'previous', 'next']:
            if ref in data:
                image_ref = data[ref]
                image_stack = torch.cat([image, image_ref], dim=1)
                motion = self.motion_net(image_stack)
                if self.rotation_scale > 0.0:
                    motion[:, 0:3] *= self.rotation_scale
                else:
                    motion[:, 0:3] = torch.zeros_like(motion[:, 0:3])
                if self.translation_scale > 0.0:
                    motion[:, 3:6] *= self.translation_scale
                else:
                    motion[:, 3:6] = torch.zeros_like(motion[:, 3:6])
                data_out['motion_' + ref] = motion

        return data_out
