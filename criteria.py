
import torch.nn
import util


class Criteria(torch.nn.Module):
    def __init__(self, depth_weight=1.0, regular_weight=1.0,
                 ref_weight=1.0, ground_weight=1.0, scale_weight=1.0, down_times=4,
                 scale_epsilon=1.0, occlusion=True):
        super(Criteria, self).__init__()
        self.regular_weight = regular_weight
        self.ref_weight = ref_weight
        self.depth_weight = depth_weight
        self.ground_weight = ground_weight
        self.scale_weight = scale_weight
        self.down_times = down_times
        self.scale_epsilon = scale_epsilon
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

        if self.scale_weight > 0.0 and 'distance_previous' in data_in and 'distance_previous' in data_in:
            motion = data_out['motion']
            scale = data_out['scale']
            data_out['eval_scale'] = scale.mean()
            # distance previous
            dp_in = data_in['distance_previous']
            mask = dp_in > 0.25
            if torch.any(mask):
                dp_out = torch.norm(motion[:, 3:6].detach(), dim=1, keepdim=True)
                loss_scale_previous = (1.0 - dp_out[mask] * scale[mask] / dp_in[mask]).abs().mean()
                data_out['loss_scale_previous'] = loss_scale_previous * self.scale_weight
                loss += data_out['loss_scale_previous']
                data_out['eval_dp_in'] = dp_in[mask].mean()
                data_out['eval_dp_out'] = dp_out[mask].mean()
            # distance next
            dn_in = data_in['distance_next']
            mask = dn_in > 0.25
            if torch.any(mask):
                dn_out = torch.norm(motion[:, 9:12].detach(), dim=1, keepdim=True)
                loss_scale_next = (1.0 - dn_out[mask] * scale[mask] / dn_in[mask]).abs().mean()
                data_out['loss_scale_next'] = loss_scale_next * self.scale_weight
                loss += data_out['loss_scale_next']
                data_out['eval_dn_in'] = dn_in[mask].mean()
                data_out['eval_dn_out'] = dn_out[mask].mean()

        if self.ground_weight > 0.0:
            ground_grid = util.plane_grid(ground.detach(), camera.detach(), image.shape, image.device)
            data_out['ground_grid'] = ground_grid
            ground_normal = ground[:, 0:3].reshape(-1, 3, 1, 1)
            ground_d = -ground[:, 3:4].reshape(-1, 1, 1, 1)
            normal_weight = normal.detach() - ground_normal.detach()
            normal_weight = torch.pow(normal_weight, 2.0).sum(dim=1, keepdim=True)
            normal_weight = torch.exp(-normal_weight * 5.0)
            data_out['ground_normal_weight'] = normal_weight

            ground_for_dist = torch.cat([ground_normal.detach(), ground_d], dim=1)
            ground_dist = util.plane_dist(ground_for_dist, camera.detach(), depth_out.detach()).abs()
            data_out['ground_dist'] = ground_dist
            dist_weight = torch.pow(ground_dist.detach(), 2.0)
            dist_weight = torch.exp(-25.0 * dist_weight)
            data_out['ground_dist_weight'] = dist_weight

            ground_weight = normal_weight * dist_weight
            data_out['ground_weight'] = ground_weight

            ground_normal_residual = ground_weight * (ground_normal - normal.detach()).abs()
            data_out['ground_normal_residual'] = ground_normal_residual.mean(dim=1, keepdim=True)
            data_out['loss_ground_normal'] = data_out['ground_normal_residual'].mean()
            loss += data_out['loss_ground_normal']

            ground_dist_residual = ground_weight * ground_dist.abs()
            data_out['ground_dist_residual'] = ground_dist_residual
            data_out['loss_ground_dist'] = data_out['ground_dist_residual'].mean()
            loss += data_out['loss_ground_dist']

        if self.regular_weight > 0.0:
            loss_regular = 0.0
            depth_down = depth_out
            image_down = image
            for i in range(min(self.down_times, 1)):
                scale_factor = 1.0 if i == 0 else 0.5
                image_down = torch.nn.functional.interpolate(
                        image_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                depth_down = torch.nn.functional.interpolate(
                            depth_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)

                image_grad = torch.cat(util.grad(image_down), dim=1).abs().mean(dim=1, keepdim=True)
                image_grad = torch.cat(util.grad(image_grad), dim=1).abs().mean(dim=1, keepdim=True)

                depth_grad = torch.cat(util.grad(depth_down), dim=1).abs().mean(dim=1, keepdim=True)
                depth_grad = torch.cat(util.grad(depth_grad), dim=1).abs().mean(dim=1, keepdim=True)

                regular_weight = torch.exp(-400.0 * image_grad * image_grad)
                regular_residual = regular_weight * depth_grad
                # regular_residual = depth_grad

                data_out['regular_image_down_%d' % i] = image_down
                data_out['regular_depth_down_%d' % i] = depth_down
                data_out['regular_image_grad_%d' % i] = image_grad
                data_out['regular_depth_grad_%d' % i] = depth_grad
                data_out['regular_weight_%d' % i] = regular_weight
                data_out['regular_residual_%d' % i] = regular_residual
                loss_regular += regular_residual.mean()

            data_out['loss_regular'] = loss_regular / min(self.down_times, 1) * self.regular_weight
            loss += data_out['loss_regular']

        if 'depth' in data_in:
            depth_in = data_in['depth']
            depth_out = data_out['depth']
            scale = data_out['scale'].reshape(-1, 1, 1, 1)
            mask = depth_in > (1.0 / 80.0)
            mask &= depth_out > (1.0 / 80.0)
            z_in = torch.zeros_like(depth_in)
            z_out = torch.zeros_like(depth_out)
            z_in[mask] = 1.0 / depth_in[mask]
            z_out[mask] = 1.0 / depth_out[mask]

            residual_abs_rel = torch.zeros_like(depth_in)
            residual_abs_rel[mask] = (1.0 - z_out[mask] / z_in[mask]).abs()
            data_out['residual_abs_rel'] = residual_abs_rel
            data_out['eval_abs_rel'] = residual_abs_rel.sum() / mask.sum()

            residual_abs_rel_scaled = torch.zeros_like(depth_in)
            residual_abs_rel_scaled[mask] = (1.0 - (z_out * scale)[mask] / z_in[mask]).abs()
            data_out['residual_abs_rel_scaled'] = residual_abs_rel_scaled
            data_out['eval_abs_rel_scaled'] = residual_abs_rel_scaled.sum() / mask.sum()
            data_out['eval_scale'] = scale.mean()

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
                warp_previous, record_previous = util.warp(
                    image_down, depth_down, camera, motion_previous, self.occlusion)
                warp_next, record_next = util.warp(
                    image_down, depth_down, camera, motion_next, self.occlusion)
                mask_previous = record_previous != 0.0
                mask_next = record_next != 0.0
                residual_previous = (image_previous_down * mask_previous - warp_previous).abs()
                residual_next = (image_next_down * mask_next - warp_next).abs()
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


