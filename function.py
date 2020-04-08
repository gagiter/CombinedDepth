
import torch
import libwarp


class warp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, grid, occlusion=None):
        if occlusion is None:
            warped = libwarp.forward(image, grid)
            ctx.save_for_backward(image, grid, occlusion)
            return warped
        else:
            return None

    @staticmethod
    def backward(ctx, grad_output):
        image, grid, occlusion = ctx.saved_tensors
        grad_image = grad_grid = grad_occlusion = None
        if occlusion is None and ctx.needs_input_grad[1]:
            grad_grid = libwarp.backward(image, grid, grad_output)

        return grad_image, grad_grid, grad_occlusion


if __name__ == '__main__':

    image = torch.rand([2, 3, 128, 256], dtype=torch.float32, device='cuda:0')
    grid = torch.rand([2, 2, 128, 256], dtype=torch.float32, device='cuda:0')
    warped = libwarp.forward(image, grid)
    help(libwarp)
    help(libwarp.forward)