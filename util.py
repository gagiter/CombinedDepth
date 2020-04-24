
import torch
import torch.nn.functional as F
import function
import math


def hit_plane(ground, camera, image):
    batch, _, height, width = image.shape
    uv = near_grid(camera, [height, width], image.device)
    ones = torch.ones([batch, 1, height, width], dtype=uv.dtype, device=uv.device)
    ray = torch.cat([uv, ones], dim=1)
    hit = ground[:, 0:3].reshape(batch, 3, 1, 1)
    hit = (hit * ray).sum(dim=1, keepdim=True)
    hit = hit / (ground[:, 3].reshape(batch, 1, 1, 1) + 0.00001)
    mask = hit < 0.0
    mask |= hit > 1.0
    hit[mask] = 0.0
    return hit


def normal(depth, camera):
    points = unproject(depth, camera)
    points = points[:, 0:3, ...] / (points[:, 3:4, ...] + 0.00001)
    grad_x, grad_y = sobel(points)
    normals = cross(grad_x, grad_y)
    normals = torch.nn.functional.normalize(normals)
    return normals, points


def cross(a, b):
    c = torch.zeros_like(a)
    c[:, 0, ...] = a[:, 1, ...] * b[:, 2, ...] - a[:, 2, ...] * b[:, 1, ...]
    c[:, 1, ...] = a[:, 2, ...] * b[:, 0, ...] - a[:, 0, ...] * b[:, 2, ...]
    c[:, 2, ...] = a[:, 0, ...] * b[:, 1, ...] - a[:, 1, ...] * b[:, 0, ...]
    return c


def sobel(image):
    channles = image.shape[1]
    filter_x = torch.zeros([1, 1, 5, 5], dtype=torch.float, device=image.device)
    filter_x[:, :, :, 0:1] = -0.5
    filter_x[:, :, :, 1:2] = -1.0
    filter_x[:, :, :, 3:4] = +1.0
    filter_x[:, :, :, 4:5] = +0.5
    filter_y = torch.zeros([1, 1, 5, 5], dtype=torch.float, device=image.device)
    filter_y[:, :, 0:1, :] = -0.5
    filter_y[:, :, 1:2, :] = -1.0
    filter_y[:, :, 3:4, :] = +1.0
    filter_y[:, :, 4:5, :] = +0.5
    filter_x = filter_x.repeat(channles, 1, 1, 1)
    filter_y = filter_y.repeat(channles, 1, 1, 1)

    grad_x = F.conv2d(image, filter_x, padding=2, groups=channles)
    grad_y = F.conv2d(image, filter_y, padding=2, groups=channles)
    return grad_x / 7.5, grad_y / 7.5


def color_map(image):
    if image.shape[-3] != 1:
        return image
    red = torch.sin(image * math.pi - math.pi * 0.0)
    green = torch.sin(image * math.pi * 5.0 - math.pi * 0.0)
    blue = torch.sin(image * math.pi * 9.0 - math.pi * 0.0)

    red = red * 0.5 + 0.5
    green = green * 0.5 + 0.5
    blue = blue * 0.5 + 0.5
    return torch.cat([red, green, blue], dim=-3)


def visualize(data):
    if 'image' in data:
        batch, _, height, width = data['image'].shape
        cm = torch.linspace(0.0, 1.0, height, dtype=torch.float)
        cm = cm.repeat([batch, 1, width, 1]).permute(0, 1, 3, 2)
        data['color_map'] = color_map(cm)
    if 'depth' in data:
        data['depth_v'] = color_map(data['depth'])
    if 'normal' in data:
        data['normal_v'] = data['normal'] * 0.5 + 0.5


def camera_scope(camera, shape):
    half_fov = 0.25 + camera[:, 0]  # mean half_fov_y = 0.42 rad about 25 deg
    center_x = camera[:, 1]
    center_y = camera[:, 2]
    if shape[0] > shape[1]:  # portrait
        half_width = torch.tan(half_fov)
        half_height = half_width * (shape[0] / shape[1])
    else:  # landscape
        half_height = torch.tan(half_fov)
        half_width = half_height * (shape[1] / shape[0])

    left = center_x - half_width
    top = center_y - half_height
    width = half_width * 2.0
    height = half_height * 2.0
    return torch.stack([left, top, width, height], dim=-1)


def near_grid(camera, shape, device):
    scope = camera_scope(camera, shape)
    grid_x = torch.linspace(-1.0, 1.0, shape[1], device=device)
    grid_x = grid_x[None, ...]
    grid_x = grid_x.repeat([camera.shape[0], 1])
    grid_x = 0.5 * grid_x * scope[..., 2:3] + scope[..., 0:1] + 0.5 * scope[..., 2:3]
    grid_x = grid_x.unsqueeze(1)
    grid_x = grid_x.repeat([1, shape[0], 1])

    grid_y = torch.linspace(-1.0, 1.0, shape[0], device=device)
    grid_y = grid_y[None, ...]
    grid_y = grid_y.repeat([camera.shape[0], 1])
    grid_y = 0.5 * grid_y * scope[..., 3:4] + scope[..., 1:2] + 0.5 * scope[..., 3:4]
    grid_y = grid_y.unsqueeze(-1)
    grid_y = grid_y.repeat([1, 1, shape[1]])

    grid = torch.stack([grid_x, grid_y], dim=1)
    return grid


def unproject(depth, camera):
    grid = near_grid(camera, depth.shape[-2:], device=depth.device)
    points = torch.cat([grid, torch.ones_like(depth), depth], dim=1)
    return points


def sample(image, uv, camera, depth):
    scope = camera_scope(camera, image.shape[-2:])
    uv = (uv - scope[..., 0:2, None, None]) / scope[..., 2:4, None, None]
    uv = 2.0 * uv - 1.0
    sampled = function.WarpFuncion.apply(image, uv, depth)

    return sampled


def warp(image, depth, camera, motion, occlusion=True):
    points = unproject(depth, camera)
    uv = project(points, camera, motion)
    warped = sample(image, uv, camera, depth if occlusion else None)
    return warped


def transfrom_matrix(motion):

    zeros = torch.zeros([motion.shape[0]], dtype=motion.dtype, device=motion.device)
    ones = torch.ones([motion.shape[0]], dtype=motion.dtype, device=motion.device)

    bank = motion[..., 0]
    heading = motion[..., 1]
    altitude = motion[..., 2]
    tx = motion[..., 3]
    ty = motion[..., 4]
    tz = motion[..., 5]
    
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

    m0 = torch.stack([r00, r01, r02, tx], dim=-1)
    m1 = torch.stack([r10, r11, r12, ty], dim=-1)
    m2 = torch.stack([r20, r21, r22, tz], dim=-1)
    m3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    m = torch.stack([m0, m1, m2, m3], dim=1)

    return m


def transform(motion, points):
    matrix = transfrom_matrix(motion)
    B, C, H, W = points.shape
    points = points.view([B, C, -1])
    points = torch.bmm(matrix, points)
    points = points.view([B, C, H, W])
    return points


def distort(uv, coefficients):
    k1 = coefficients[..., 0:1]
    k2 = coefficients[..., 1:2]
    r2 = torch.pow(uv, 2).sum(dim=1, keepdim=True)
    r4 = r2 * r2
    distorted = uv * (1.0 + k1[..., None, None] * r2 + k2[..., None, None] * r4)
    return distorted


def project(points, camera, motion):
    points = transform(motion, points)
    uv = points[:, 0:2, ...]
    uv = uv / points[:, 2:3, ...]
    uv = distort(uv, camera[..., 3:5])
    return uv


if __name__ == '__main__':
    image = torch.rand([2, 3, 128, 256], device='cuda:0')
    depth = torch.rand([2, 1, 128, 256], device='cuda:0')
    camera = torch.randn([2, 6], device='cuda:0') * 0.01
    motion = torch.randn([2, 6], device='cuda:0') * 0.01
    w = warp(image, depth, camera, motion)
