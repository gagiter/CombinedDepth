
import torch
import libwarp


class warp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, grid, occlusion=None):
        warped = libwarp.forward(image, grid, occlusion)
        ctx.save_for_backward(image, grid, occlusion)
        return warped

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        grad = libwarp.backward(ctx)
        return grad
