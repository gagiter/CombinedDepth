
import torch.nn


class Criteria(torch.nn.Module):
    def __init__(self):
        super(Criteria, self).__init__()

    def forward(self, data_in, data_out):
        loss = 0.0
        if 'depth' in data_in:
            depth_in = data_in['depth']
            mask = data_in['mask']
            depth_out = data_out['depth']
            loss_depth = ((depth_in - depth_out) * mask).abs().sum() / mask.sum()
            loss += loss_depth

        for ref in ['stereo', 'previous', 'next']:
            if ref in data_in:
                image = data_in['image']
                image_ref = data_in[ref]
                depth = data_out['depth'] * 80.0
                camera = data_out['camera']
                motion_stereo = data_out['motion' + ref]
                warp_ref = util.warp(image_ref, depth, camera, motion_stereo)
                loss_ref = ((image - warp_ref) * mask).abs().sum() / mask.sum()
                loss += loss_ref

        return loss

