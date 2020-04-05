
import torch


def colorize(data):
    depth = data['depth']
    red = depth
    green = 1.0 - depth
    blue = depth  # torch.zeros_like(depth)
    color = torch.cat([red, green, blue], dim=1)
    data['depth_c'] = color


def warp(image, depth, camera, motion):
    points = unproject(depth, camera)
    uvs = project(points, camera, motion)
    warped = warp(image, uvs, depth)
    return warped


def unproject(depth, camera):
    near = near_grid(camera, depth.shape[-2:])
    near = undistort(near, camera)
    points = near * depth
    return points


def project(points, camera, motion):
    points = transform(motion, points)
    k = camera_matrix(camera)
    uvs = k * points
    uvs = distort(uvs, camera)
    return uvs


def distort(uvs, camera):
    return None
