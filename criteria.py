
import torch.nn
import util


class Criteria(torch.nn.Module):
    def __init__(self):
        super(Criteria, self).__init__()

    def forward(self, data_in, data_out):
        loss = 0.0
        depth_out = data_out['depth']
        if 'depth' in data_in:
            depth_in = data_in['depth']
            mask = data_in['mask']
            loss_depth = ((depth_in - depth_out) * mask).abs().sum() / mask.sum()
            loss += loss_depth

        for ref in ['stereo', 'previous', 'next']:
            if ref in data_in:
                image = data_in['image']
                image_ref = data_in[ref]
                # depth = data_out['depth'] * 80.0
                camera = data_out['camera']
                motion = data_out['motion_' + ref]
                warp = util.warp(image_ref, depth_out, camera, motion)
                loss_ref = ((image - warp) * mask).abs().sum() / mask.sum()
                # loss += loss_ref

        return loss

