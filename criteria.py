
import torch.nn
import util


class Criteria(torch.nn.Module):
    def __init__(self, depth_weight=1.0, regular_weight=1.0,
                 ref_weight=1.0, ground_weight=1.0, down_times=4, occlusion=True,
                 global_depth=1):
        super(Criteria, self).__init__()
        self.regular_weight = regular_weight
        self.ref_weight = ref_weight
        self.depth_weight = depth_weight
        self.ground_weight = ground_weight
        self.down_times = down_times
        self.occlusion = occlusion
        self.global_depth = global_depth

    def forward(self, data_in, data_out):
        loss = 0.0
        image = data_in['image']
        camera = data_out['camera']
        ground = data_out['ground']
        depth_out = data_out['depth']
        normal, points = util.normal(depth_out, camera)
        data_out['normal'] = normal
        data_out['points'] = points

        if self.ground_weight > 0.0:
            hit = util.hit_plane(ground, camera, image)
            # ground_residual = ground[:3].dot(normal) * (hit - depth_out).abs()
            data_out['ground_hit'] = hit
            ground_dot = ground[:, 0:3].reshape(-1, 3, 1, 1)
            ground_dot = (ground_dot * normal).sum(dim=1, keepdim=True)
            data_out['ground_dot'] = ground_dot
            ground_dist = torch.exp(-2.8 * torch.pow((hit - depth_out), 2))
            data_out['ground_dist'] = ground_dist
            data_out['ground_residual'] = 1.0 - (ground_dot * ground_dist)
            data_out['loss_ground'] = data_out['ground_residual'].mean()
            loss += data_out['loss_ground']

        if self.regular_weight > 0.0:
            loss_regular = 0.0
            depth_down = depth_out
            image_down = image
            normal_down = normal
            for i in range(self.down_times):
                scale_factor = 1.0 if i == 0 else 0.5
                image_down = torch.nn.functional.interpolate(
                        image_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                normal_down = torch.nn.functional.interpolate(
                        normal_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                normal_grad = torch.cat(util.sobel(normal_down), dim=1).abs().mean(dim=1, keepdim=True)
                image_grad = torch.cat(util.sobel(image_down), dim=1).abs().mean(dim=1, keepdim=True)
                image_grad_inv = 1.0 - image_grad
                regular = normal_grad * image_grad_inv
                data_out['grad_normal_%d' % i] = normal_grad
                data_out['grad_image_%d' % i] = image_grad
                data_out['grad_image_inv_%d' % i] = image_grad_inv
                data_out['grad_regular_%d' % i] = regular
                loss_regular += regular.mean()

            data_out['loss_regular'] = loss_regular / self.down_times * self.regular_weight
            loss += data_out['loss_regular']

        if 'depth' in data_in:
            depth_in = data_in['depth']
            depth_out = data_out['depth']
            mask = depth_in > (1.0 / 100.0)
            residual_depth = torch.zeros_like(depth_in)
            residual_depth[mask] = depth_in[mask] - depth_out[mask]
            abs_rel = 1.0 - (depth_in[mask]/depth_out[mask])
            data_out['abs_rel'] = abs_rel.abs().mean()
            if self.depth_weight > 0.0:
                data_out['residual_depth'] = residual_depth.abs()
                data_out['abs_rel_global'] = \
                    (1.0 - depth_in[mask].mean() / depth_out[mask].mean()).abs()
                if self.global_depth:
                    data_out['loss_depth'] = data_out['abs_rel_global'] * self.depth_weight
                else:
                    data_out['loss_depth'] = data_out['abs_rel'] * self.depth_weight
                loss += data_out['loss_depth']

        for ref in ['stereo', 'previous', 'next']:
            if ref in data_in and self.ref_weight > 0.0:
                image_ref = data_in[ref]
                motion = data_out['motion_' + ref]
                data_out['base_' + ref] = (image - image_ref).abs()
                loss_ref = 0.0
                image_down = image
                image_ref_down = image_ref
                depth_down = depth_out
                for i in range(self.down_times):
                    scale_factor = 1.0 if i == 0 else 0.5
                    image_down = torch.nn.functional.interpolate(
                        image_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                    image_ref_down = torch.nn.functional.interpolate(
                        image_ref_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                    depth_down = torch.nn.functional.interpolate(
                        depth_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                    warp = util.warp(image_ref_down, depth_down, camera, motion, self.occlusion)
                    mask = warp > 0.0
                    residual = (image_down * mask - warp).abs()
                    data_out['residual_%s_%d' % (ref, i)] = residual
                    data_out['warp_%s_%d' % (ref, i)] = warp
                    loss_ref += residual.sum() / (mask.sum() + 1)
                data_out['loss_' + ref] = loss_ref / self.down_times * self.ref_weight
                loss += data_out['loss_' + ref]

        return loss

