
import torch.nn
import util


class Criteria(torch.nn.Module):
    def __init__(self, depth_weight=1.0, regular_weight=1.0,
                 ref_weight=1.0, ground_weight=1.0, down_times=4, occlusion=True):
        super(Criteria, self).__init__()
        self.regular_weight = regular_weight
        self.ref_weight = ref_weight
        self.depth_weight = depth_weight
        self.ground_weight = ground_weight
        self.down_times = down_times
        self.occlusion = occlusion

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
            for i in range(self.down_times):
                scale_factor = 1.0 if i == 0 else 0.5
                image_down = torch.nn.functional.interpolate(
                        image_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                depth_down = torch.nn.functional.interpolate(
                            depth_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                normal_down, _ = util.normal(depth_down, camera)
                normal_grad = torch.cat(util.sobel(normal_down), dim=1)\
                    .abs().mean(dim=1, keepdim=True)
                image_grad = torch.cat(util.sobel(image_down), dim=1)\
                    .abs().mean(dim=1, keepdim=True)
                image_grad_inv = torch.exp(-8.0 * image_grad * image_grad)
                regular = normal_grad * image_grad_inv
                data_out['grad_depth_down_%d' % i] = depth_down
                data_out['grad_normal_down_%d' % i] = normal_down * 0.5 + 0.5
                data_out['grad_normal_%d' % i] = normal_grad
                data_out['grad_image_inv_%d' % i] = image_grad_inv
                data_out['grad_regular_%d' % i] = regular
                loss_regular += regular.mean()

            data_out['loss_regular'] = loss_regular / self.down_times * self.regular_weight
            loss += data_out['loss_regular']

        if 'depth' in data_in:
            depth_in = data_in['depth']
            depth_out = data_out['depth']
            mask = depth_in > (1.0 / 50.0)
            mask &= depth_out > (1.0 / 50.0)

            residual_abs_rel = torch.zeros_like(depth_in)
            residual_abs_rel[mask] = (1.0 - depth_in[mask] / depth_out[mask]).abs()
            data_out['residual_abs_rel'] = residual_abs_rel
            data_out['eval_abs_rel'] = residual_abs_rel.sum() / mask.sum()

            z_in = torch.zeros_like(depth_in)
            z_out = torch.zeros_like(depth_out)
            z_in[mask] = 1.0 / depth_in[mask]
            z_out[mask] = 1.0 / depth_out[mask]
            z_in_mean = z_in.sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True)
            z_out_mean = z_out.sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True)
            global_scale = z_in_mean / z_out_mean
            depth_in *= global_scale
            residual_abs_rel_global = torch.zeros_like(depth_in)
            residual_abs_rel_global[mask] = (1.0 - depth_in[mask] / depth_out[mask]).abs()

            data_out['residual_abs_rel_global'] = residual_abs_rel_global
            data_out['eval_abs_rel_global'] = residual_abs_rel_global.sum() / mask.sum()
            data_out['eval_global_scale'] = global_scale.mean()

            if self.depth_weight > 0.0:
                data_out['loss_depth'] = data_out['eval_abs_rel'] * self.depth_weight
                loss += data_out['loss_depth']

        if self.ref_weight > 0:
            loss_previous = 0.0
            loss_next = 0.0
            data_out['base_previous'] = (image - data_in['previous']).abs()
            data_out['base_next'] = (image - data_in['next']).abs()
            data_out['image_previous'] = data_in['previous']
            data_out['image_next'] = data_in['next']
            image_down = data_in['image']
            image_previous_down = data_in['previous']
            image_next_down = data_in['next']
            depth_down = data_out['depth']
            motion_previous = data_out['motion'][:, 0:6]
            motion_next = data_out['motion'][:, 6:12]
            for i in range(self.down_times):
                scale_factor = 1.0 if i == 0 else 0.5
                image_down = torch.nn.functional.interpolate(
                    image_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                image_previous_down = torch.nn.functional.interpolate(
                    image_previous_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                image_next_down = torch.nn.functional.interpolate(
                    image_next_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                depth_down = torch.nn.functional.interpolate(
                    depth_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                warp_previous = util.warp(image_previous_down, depth_down, camera, motion_previous, self.occlusion)
                warp_next = util.warp(image_next_down, depth_down, camera, motion_next, self.occlusion)
                mask_previous = warp_previous > 0.0
                mask_next = warp_next > 0.0
                residual_previous = (image_down * mask_previous - warp_previous).abs()
                residual_next = (image_down * mask_next - warp_next).abs()
                data_out['residual_previous_%d' % i] = residual_previous
                data_out['residual_next_%d' % i] = residual_next
                data_out['warp_previous_%d' % i] = warp_previous
                data_out['warp_next_%d' % i] = warp_next
                loss_previous += residual_previous.sum() / (mask_previous.sum() + 1)
                loss_next += residual_next.sum() / (mask_next.sum() + 1)
            data_out['loss_previous'] = loss_previous / self.down_times * self.ref_weight
            data_out['loss_next'] = loss_next / self.down_times * self.ref_weight
            loss += data_out['loss_previous'] + data_out['loss_next']

        return loss


