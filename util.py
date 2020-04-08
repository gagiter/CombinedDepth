
import torch
import torch.nn.functional as F
import function


def grad(image):
    filter_x = torch.tensor([[-0.25, 0.0, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.0, 0.25]],
                            device=image.device).view(1, 1, 3, 3)
    filter_y = torch.tensor([[-0.25, -0.5, -0.25], [0.0, 0.0, 0.0], [0.25, 0.5, 0.25]],
                            device=image.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(image, filter_x, padding=1)
    grad_y = F.conv2d(image, filter_y, padding=1)
    return torch.cat([grad_x, grad_y], dim=1)


def visualize(data):
    depth = data['depth']
    depth = torch.exp(-0.025 * depth)
    depth_i = 1.0 - depth
    color = torch.cat([depth_i, depth_i, depth], dim=1)
    data['depth_v'] = color


def camera_scope(camera, shape):
    half_fov_y = 0.42 + camera[:, 0]  # mean half_fov_y = 0.42 rad about 25 deg
    half_fov_x = half_fov_y * (1.0 + camera[:, 1]) * (shape[1] / shape[0])
    center_x = camera[:, 2]
    center_y = camera[:, 3]
    half_width = torch.tan(half_fov_x)
    half_height = torch.tan(half_fov_y)
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


def warp(image, depth, camera, motion):
    points = unproject(depth, camera)
    uv = project(points, camera, motion)
    warped = sample(image, uv, camera, depth)
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
