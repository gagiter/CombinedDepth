
import torch
import torch.nn.functional as F
import function
import math
import numpy as np
from PIL.ExifTags import TAGS, GPSTAGS


def plane_dist(plane, camera, depth):
    points = unproject(depth, camera)
    dist = (points * plane).sum(dim=1, keepdim=True)
    return dist


def plane_grid(ground, camera, shape, device):
    batch, _, height, width = shape
    uv = near_grid(camera, [height, width], device)
    ones = torch.ones([batch, 1, height, width], dtype=uv.dtype, device=uv.device)
    ray = torch.cat([uv, ones], dim=1)
    ground_y = ground[:, 0:3].reshape(batch, 3, 1, 1)
    ground_x = torch.zeros_like(ground_y)
    ground_x[:, 0, ...] = 1.0
    ground_z = cross(ground_x, ground_y)
    ground_z = torch.nn.functional.normalize(ground_z)
    ground_x = cross(ground_z, ground_y)
    rn = (ray * ground_y).sum(dim=1, keepdim=True)
    d = ground[:, 3].reshape(batch, 1, 1, 1)
    z = d / rn
    p = z * ray
    p_x = (p * ground_x).sum(dim=1, keepdim=True)
    p_z = (p * ground_z).sum(dim=1, keepdim=True)
    p_xz = torch.cat([p_x, p_z], dim=1)
    grid = torch.pow(p_xz - torch.round(p_xz), 2.0).min(dim=1, keepdim=True)[0]
    grid = torch.exp(-100.0 * grid)
    grid[p_z < 0.0] = 0.0
    grid[p_z > 80.0] = 0.0
    return grid


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
    grad_x, grad_y = sobel(points, padding=1)
    normals = cross(grad_x, grad_y)
    normals = torch.nn.functional.normalize(normals)
    return normals, points


def cross(a, b):
    c = torch.zeros_like(a)
    c[:, 0, ...] = a[:, 1, ...] * b[:, 2, ...] - a[:, 2, ...] * b[:, 1, ...]
    c[:, 1, ...] = a[:, 2, ...] * b[:, 0, ...] - a[:, 0, ...] * b[:, 2, ...]
    c[:, 2, ...] = a[:, 0, ...] * b[:, 1, ...] - a[:, 1, ...] * b[:, 0, ...]
    return c


def grad(image, padding=0):
    channles = image.shape[-3]
    filter_x = np.array([[0.0, -0.25, 0.25], [0.0, -0.5, 0.5], [0.0, -0.25, 0.25]],
                        dtype=np.float32)
    filter_x = torch.from_numpy(filter_x).to(image.device).reshape(1, 1, 3, 3)
    filter_y = np.array([[0.0, 0.0, 0.0], [-0.25, -0.5, -0.25], [0.25, 0.5, 0.25]],
                        dtype=np.float32)
    filter_y = torch.from_numpy(filter_y).to(image.device).reshape(1, 1, 3, 3)
    filter_x = filter_x.repeat(channles, 1, 1, 1)
    filter_y = filter_y.repeat(channles, 1, 1, 1)
    grad_x = F.conv2d(image, filter_x, padding=padding, groups=channles)
    grad_y = F.conv2d(image, filter_y, padding=padding, groups=channles)
    return grad_x, grad_y


def sobel(image, padding=0):
    channles = image.shape[-3]
    filter_x = np.array([[-0.25, 0.0, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.0, 0.25]],
                        dtype=np.float32)
    filter_x = torch.from_numpy(filter_x).to(image.device).reshape(1, 1, 3, 3)
    filter_y = np.array([[-0.25, -0.5, -0.25], [0.0, 0.0, 0.0], [0.25, 0.5, 0.25]],
                        dtype=np.float32)
    filter_y = torch.from_numpy(filter_y).to(image.device).reshape(1, 1, 3, 3)
    filter_x = filter_x.repeat(channles, 1, 1, 1)
    filter_y = filter_y.repeat(channles, 1, 1, 1)
    grad_x = F.conv2d(image, filter_x, padding=padding, groups=channles)
    grad_y = F.conv2d(image, filter_y, padding=padding, groups=channles)
    return grad_x, grad_y


def color_map(image):
    if image.shape[-3] != 1:
        return image
    red = torch.sin(image * math.pi - math.pi * 0.5)
    green = torch.sin(image * math.pi * 2.0 - math.pi * 0.0)
    blue = torch.sin(image * math.pi * 4.0 - math.pi * 0.0)

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


def sample(image, uv, camera, depth, warp_flag, record_sigma=0.5):
    scope = camera_scope(camera, image.shape[-2:])
    uv = (uv - scope[..., 0:2, None, None]) / scope[..., 2:4, None, None]
    uv = 2.0 * uv - 1.0
    if warp_flag == 0:
        return function.warp.apply(image, uv)
    elif warp_flag == 1:
        return function.warp_direct.apply(image, uv, depth)
    elif warp_flag == 2:
        sigma_tensor = torch.ones(1, dtype=image.dtype, device=image.device) * record_sigma
        return function.warp_record.apply(image, uv, depth, sigma_tensor)
    elif warp_flag == 3:
        return function.warp_wide.apply(image, uv)
    else:
        raise Exception('Invalid warp flag.')


def warp(image, depth, camera, motion, warp_flag=0, record_sigma=0.5):
    points = unproject(depth, camera)
    uv = project(points, camera, motion)
    return sample(image, uv, camera, depth, warp_flag, record_sigma)


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


class Exif:
    @classmethod
    def read_gps(cls, meta):
        items = dict()
        gps_items = dict()
        lla = [0.0, 0.0, 0.0]
        if meta is not None:
            for key, value in meta.items():
                if key in TAGS:
                    items[TAGS[key]] = meta[key]

            if 'GPSInfo' in items:
                for key in items['GPSInfo'].keys():
                    name = GPSTAGS[key]
                    gps_items[name] = items['GPSInfo'][key]

                for index, name in enumerate(['GPSLatitude', 'GPSLongitude', 'GPSAltitude']):
                    e = gps_items[name]
                    ref = gps_items[name + 'Ref']
                    if name in ['GPSLatitude', 'GPSLongitude']:
                        lla[index] = e[0][0] / e[0][1]
                        lla[index] += e[1][0] / e[1][1] / 60
                        lla[index] += e[2][0] / e[2][1] / 3600
                        lla[0] *= -1 if ref in ['S', 'W'] else 1
                    else:
                        lla[index] = e[0] / e[1]
        return lla



class GPS:
    EarthRadius = 6378137.0
    e2 = 0.00669437999013

    def __init__(self):
        self.origin = [0.0, 0.0, 0.0]
        self.has_origin = False

    def set_origin(self, latitude, longitude, altitude):

        sin_latitude = math.sin(math.pi * latitude / 180.0)
        cos_latitude = math.cos(math.pi * latitude / 180.0)
        sin_longitude = math.sin(math.pi * longitude / 180.0)
        cos_longitude = math.cos(math.pi * longitude / 180.0)
        N = self.EarthRadius /\
                 (math.sqrt(1.0 - self.e2 * math.pow(sin_latitude, 2)))

        self.origin[0] = (N + altitude) * cos_latitude * cos_longitude
        self.origin[1] = (N + altitude) * cos_latitude * sin_longitude
        self.origin[2] = (N + (1.0 - self.e2) + altitude) * sin_latitude
        self.has_origin = True

    def lla2enu(self, latitude, longitude, altitude):

        if not self.has_origin:
            self.set_origin(latitude, longitude, altitude)
        sin_latitude = math.sin(math.pi * latitude / 180.0)
        cos_latitude = math.cos(math.pi * latitude / 180.0)
        sin_longitude = math.sin(math.pi * longitude / 180.0)
        cos_longitude = math.cos(math.pi * longitude / 180.0)
        N = self.EarthRadius / \
                 (math.sqrt(1.0 - self.e2 * math.pow(sin_latitude, 2)))

        X = [0.0, 0.0, 0.0]
        X[0] = (N + altitude) * cos_latitude * cos_longitude
        X[1] = (N + altitude) * cos_latitude * sin_longitude
        X[2] = (N + (1.0 - self.e2) + altitude) * sin_latitude

        enu = [0.0, 0.0, 0.0]

        enu[0] = -sin_longitude * (X[0] - self.origin[0]) + cos_longitude * (X[1] - self.origin[1])
        enu[1] = -sin_latitude * cos_longitude * (X[0] - self.origin[0]) - \
            sin_longitude * sin_latitude * (X[1] - self.origin[1]) + \
            cos_latitude * (X[2] - self.origin[2])
        enu[2] = cos_latitude * cos_longitude * (X[0] - self.origin[0]) + \
            cos_latitude * sin_longitude * (X[1] - self.origin[1]) + \
            sin_latitude * (X[2] - self.origin[2])

        return enu


if __name__ == '__main__':
    image = torch.rand([2, 3, 128, 256], device='cuda:0')
    depth = torch.rand([2, 1, 128, 256], device='cuda:0')
    camera = torch.randn([2, 6], device='cuda:0') * 0.01
    motion = torch.randn([2, 6], device='cuda:0') * 0.01
    w = warp(image, depth, camera, motion)
