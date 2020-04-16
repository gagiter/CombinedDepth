
import torch.nn
import util


class Criteria(torch.nn.Module):
    def __init__(self, depth_weight=1.0, normal_weight=1.0,
                 plane_weight=1.0, ref_weight=1.0, down_times=4, occlusion=True):
        super(Criteria, self).__init__()
        self.normal_weight = normal_weight
        self.plane_weight = plane_weight
        self.ref_weight = ref_weight
        self.depth_weight = depth_weight
        self.down_times = down_times
        self.occlusion = occlusion

    def forward(self, data_in, data_out):
        loss = 0.0
        image = data_in['image']
        image_grad = torch.cat(util.grad(image), dim=1)
        data_out['image_grad'] = image_grad.abs().mean(dim=1, keepdims=True)
        camera = data_out['camera']
        depth_out = data_out['depth']
        normals, points = util.normal(depth_out, camera)
        data_out['normals'] = normals
        data_out['points'] = points

        if self.normal_weight > 0.0:
            loss_normal = 0.0
            depth_down = depth_out
            for i in range(self.down_times):
                depth_down = torch.nn.functional.interpolate(
                        depth_down, scale_factor=0.5, mode='bilinear', align_corners=True)
                normals, _ = util.normal(depth_down, camera)
                normals_grad = torch.cat(util.grad(normals), dim=1)
                data_out['residual_normals_%d' % i] = normals * 0.5 + 0.5
                data_out['residual_normals_grad_%d' % i] = normals_grad.abs().mean(dim=1, keepdims=True)
                loss_normal += data_out['residual_normals_grad_%d' % i].mean()
            data_out['loss_normal'] = loss_normal / self.down_times * self.normal_weight
            loss += data_out['loss_normal']

        if self.plane_weight > 0.0:
            loss_plane = 0.0
            depth_down = depth_out
            for i in range(self.down_times):
                scale_factor = 1.0 if i == 0 else 0.5
                depth_down = torch.nn.functional.interpolate(
                    depth_down, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                normals, points = util.normal(depth_down, camera)
                plane = (points * normals).sum(dim=1, keepdims=True)
                plane_grad = torch.cat(util.grad(plane), dim=1)
                data_out['residual_plane_%d' % i] = plane.abs()
                data_out['residual_plane_grad_%d' % i] = plane_grad.abs().mean(dim=1, keepdims=True)
                loss_plane += data_out['residual_plane_grad_%d' % i].mean()
            data_out['loss_plane'] = loss_plane / self.down_times * self.plane_weight
            loss += data_out['loss_plane']

        if 'depth' in data_in:
            depth_in = data_in['depth']
            depth_out = data_out['depth']
            mask = depth_in > (1.0 / 80.0)
            residual_depth = torch.zeros_like(depth_in)
            residual_depth[mask] = depth_in[mask] - depth_out[mask]
            data_out['residual_depth'] = residual_depth.abs()
            abs_rel = 1.0 - (depth_in[mask]/depth_out[mask])
            data_out['abs_rel'] = abs_rel.abs().mean()
            data_out['loss_depth'] = data_out['abs_rel'] * self.depth_weight
            loss += data_out['loss_depth']

        for ref in ['stereo', 'previous', 'next']:
            if ref in data_in and self.ref_weight > 0.0:
                image_ref = data_in[ref]
                motion = data_out['motion_' + ref]
                data_out['residual_' + ref] = (image - image_ref).abs()
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
                    residual = (image_down - warp).abs()
                    data_out['residual_%s_%d' % (ref, i)] = residual
                    loss_ref += residual.mean()
                data_out['loss_' + ref] = loss_ref / self.down_times * self.ref_weight
                loss += data_out['loss_' + ref]

        return loss

