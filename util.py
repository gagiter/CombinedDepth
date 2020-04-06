
import torch


def colorize(data):
    depth = data['depth']
    red = depth
    green = 1.0 - depth
    blue = depth  # torch.zeros_like(depth)
    color = torch.cat([red, green, blue], dim=1)
    data['depth_c'] = color


def camera_scope(camera, shape):
    half_fov_y = 0.25 + camera[:, 0]
    half_fov_x = half_fov_y * (1.0 + camera[:, 1]) * (shape[1] / shape[0])
    center_x = camera[:, 2]
    center_y = camera[:, 3]
    left = center_x - torch.tan(half_fov_x)
    right = center_y + torch.tan(half_fov_x)
    top = center_y - torch.tan(half_fov_y)
    bottom = center_y + torch.tan(half_fov_y)
    return left, right, top, bottom


def near_grid(camera, shape, device):
    left, right, top, bottom = camera_scope(camera, shape)
    grid = []
    for b in range(camera.shape[0]):
        lin_x = torch.linspace(-1.0, 1.0, shape[1], device=device)
        lin_y = torch.linspace(-1.0, 1.0, shape[0], device=device)
        lin_x = 0.5 * lin_x * (right[b] - left[b]) + 0.5 * (right[b] + left[b])
        lin_y = 0.5 * lin_y * (bottom[b] - top[b]) + 0.5 * (bottom[b] + top[b])
        grid_y, grid_x = torch.meshgrid([lin_y, lin_x])
        grid.append(torch.stack([grid_x, grid_y], dim=0))

    grid = torch.stack(grid, dim=0)
    return grid


def unproject(depth, camera):
    grid = near_grid(camera, depth.shape[-2:], device=depth.device)
    points = torch.cat([grid, torch.ones_like(depth)], dim=1)
    points = points * depth
    return points


def sample(image, uv, camera, depth):
    left, right, top, bottom = camera_scope(camera, image.shape[-2:])
    left_top = torch.stack([left, top], dim=-1)
    bottom_right = torch.stack([left, top], dim=-1)
    left_top = left_top[..., None, None]
    bottom_right = bottom_right[..., None, None]
    uv = (uv - left_top) / (bottom_right - left_top)
    uv = uv.permute(0, 2, 3, 1)
    sampled = torch.nn.functional.grid_sample(
        image, uv, mode='bilinear', align_corners=True)
    return sampled


def warp(image, depth, camera, motion):
    points = unproject(depth, camera)
    uv = project(points, camera, motion)
    warped = sample(image, uv, camera, depth)
    return warped


def rotation_matrix(radians):
    bank = radians[..., 0]
    heading = radians[..., 1]
    altitude = radians[..., 2]
    #
    sa = torch.sin(altitude)
    ca = torch.cos(altitude)
    sb = torch.sin(bank)
    cb = torch.cos(bank)
    sh = torch.sin(heading)
    ch = torch.cos(heading)

    r00 = ch * ca
    r01 = sh * sb - ch * sa * cb
    r02 = ch * sa * sb + sh * cb
    r10 = sa
    r11 = ca * cb
    r12 = -ca * sb
    r20 = -sh * ca
    r21 = sh * sa * cb + ch * sb
    r22 = -sh * sa * sb + ch * cb

    r0 = torch.stack([r00, r01, r02], dim=-1)
    r1 = torch.stack([r10, r11, r12], dim=-1)
    r2 = torch.stack([r20, r21, r22], dim=-1)
    r = torch.stack([r0, r1, r2], dim=1)

    return r


def transform(motion, points):
    rotation = rotation_matrix(motion[..., 0:3])
    translation = motion[..., 3:6]
    B, C, H, W = points.shape
    points = points.view([B, C, -1])
    points = torch.bmm(rotation, points)
    points += translation[..., None]
    points = points.view([B, C, H, W])
    return points


def distort(uv, coefficients):
    k1 = coefficients[..., 0]
    k2 = coefficients[..., 1]
    r2 = torch.pow(uv, 2).sum(dim=1, keepdim=True)
    r4 = r2 * r2
    distorted = uv * (1.0 + k1[..., None, None] * r2 + k2[..., None, None] * r4)
    return distorted


def project(points, camera, motion):
    points = transform(motion, points)
    uv = points[:, 0:2, ...]
    uv = uv / points[:, 2, ...]
    uv = distort(uv, camera[..., 4:6])
    return uv


if __name__ == '__main__':
    image = torch.rand([2, 3, 128, 256], device='cuda:0')
    depth = torch.rand([2, 1, 128, 256], device='cuda:0')
    camera = torch.randn([2, 6], device='cuda:0') * 0.01
    motion = torch.randn([2, 6], device='cuda:0') * 0.01
    w = warp(image, depth, camera, motion)
