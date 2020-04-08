
import torch.nn
import util


class Criteria(torch.nn.Module):
    def __init__(self, smooth_weight=1.0, ref_weight=1.0, depth_weight=1.0, down_times=4):
        super(Criteria, self).__init__()
        self.smooth_weight = smooth_weight
        self.ref_weight = ref_weight
        self.depth_weight = depth_weight
        self.down_times = down_times

    def forward(self, data_in, data_out):
        loss = 0.0
        if self.smooth_weight > 0.0:
            depth_grad = util.grad(depth_out)
            data_out['residual_grad'] = depth_grad.abs().mean(dim=1, keepdim=True)
            data_out['loss_smooth'] = data_out['residual_grad'].mean() * self.smooth_weight
            loss += data_out['loss_smooth']

        if 'depth' in data_in:
            depth_in = data_in['depth']
            depth_out = data_out['depth']
            mask = depth_in > 0.1
            mask &= depth_in < 80.0
            depth_out = 1.0 / depth_out
            residual = depth_in - depth_out
            residual[mask] /= depth_in[mask]
            residual *= mask
            data_out['abs_rel'] = residual.abs().sum() / mask.sum()
            if self.depth_weight > 0.0:
                data_out['residual_depth'] = residual
                data_out['loss_depth'] = data_out['abs_rel'] * self.depth_weight
                loss += data_out['loss_depth']

        for ref in ['stereo', 'previous', 'next']:
            if ref in data_in and self.ref_weight > 0.0:
                depth_out = data_out['depth']
                image = data_in['image']
                image_ref = data_in[ref]
                camera = data_out['camera']
                motion = data_out['motion_' + ref]
                data_out['residual_' + ref] = (image - image_ref).abs()
                loss_ref = 0.0
                height, width = image.shape[-2:]
                for i in range(self.down_times):
                    image_down = torch.nn.functional.interpolate(
                        image, size=(height, width), mode='bilinear', align_corners=True)
                    image_ref_down = torch.nn.functional.interpolate(
                        image_ref, size=(height, width), mode='bilinear', align_corners=True)
                    depth_out_down = torch.nn.functional.interpolate(
                        depth_out, size=(height, width), mode='bilinear', align_corners=True)
                    warp = util.warp(image_ref_down, depth_out_down, camera, motion)
                    residual = (image - warp).abs()
                    data_out['residual_%s_%d' % (ref, i)] = residual
                    loss_ref += residual.mean()
                    height >>= 1
                    width >>= 1
                data_out['loss_' + ref] = loss_ref / self.down_times * self.ref_weight
                loss += data_out['loss_' + ref]

        return loss

