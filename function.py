
import torch
import libwarp
from torch.utils.tensorboard import SummaryWriter


class warp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample):
        warped = libwarp.forward(image.contiguous(), sample.contiguous())
        ctx.save_for_backward(image, sample)
        return warped

    @staticmethod
    def backward(ctx, grad_warp):
        image, sample = ctx.saved_tensors
        grad_image = grad_sample = None
        if ctx.needs_input_grad[1]:
            grad_sample = libwarp.backward(
                image.contiguous(), sample.contiguous(), grad_warp.contiguous())
        return grad_image, grad_sample


class warp_wide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample):
        warped = libwarp.forward_wide(image.contiguous(), sample.contiguous())
        ctx.save_for_backward(image, sample)
        return warped

    @staticmethod
    def backward(ctx, grad_warp):
        image, sample = ctx.saved_tensors
        grad_image = grad_sample = None
        if ctx.needs_input_grad[1]:
            grad_sample = libwarp.backward_wide(
                image.contiguous(), sample.contiguous(), grad_warp.contiguous())
        return grad_image, grad_sample


class warp_direct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample, depth):
        warped, record = libwarp.forward_direct(image.contiguous(), sample.contiguous(), depth.contiguous())
        ctx.save_for_backward(image, sample, depth, record)
        return warped, record

    @staticmethod
    def backward(ctx, grad_warp, grad_record):
        image, sample, depth, record = ctx.saved_tensors
        grad_image = grad_sample = grad_depth = None
        if ctx.needs_input_grad[1]:
            grad_sample = libwarp.backward_direct(
                image.contiguous(), sample.contiguous(), depth.contiguous(),
                record.contiguous(), grad_warp.contiguous())

        return grad_image, grad_sample, grad_depth


class warp_record(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample, depth, sigma):
        warped, record, weight_record, weight_warp = libwarp.forward_record(
            image.contiguous(), sample.contiguous(), depth.contiguous(), sigma)
        ctx.save_for_backward(image, sample, depth, record, weight_warp, sigma)
        return warped, record, weight_record, weight_warp

    @staticmethod
    def backward(ctx, grad_warp, grad_record, grad_weight_record, grad_weight_warp):
        image, sample, depth, record, weight_warp, sigma = ctx.saved_tensors
        grad_image = grad_sample = grad_depth = grad_sigma = None
        if ctx.needs_input_grad[1]:
            grad_sample = libwarp.backward_record(
                image.contiguous(), sample.contiguous(), depth.contiguous(),
                record.contiguous(), weight_warp.contiguous(), grad_warp.contiguous(), sigma)

        return grad_image, grad_sample, grad_depth, grad_sigma


def test_record(writer):
    image = torch.ones([1, 1, 100, 200], dtype=torch.float32, device='cuda:0')
    image[:, :, 20:80, 50:150] = 0.5
    ref = torch.ones([1, 1, 100, 200], dtype=torch.float32, device='cuda:0')
    ref[:, :, 30:90, 70:170] = 0.5

    depth = torch.ones([1, 1, 100, 200], dtype=torch.float32, device='cuda:0') * 0.2
    # depth[:, :, 20:80, 50:150] = 0.5

    grid_x = torch.linspace(-1.0, 1.0, 200)
    grid_y = torch.linspace(-1.0, 1.0, 100)

    grid_y, grid_x = torch.meshgrid([grid_y, grid_x])
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([1, 1, 1, 1])
    grid = grid.to('cuda:0')
    grid[:, :, 20:80, 50:150] += 0.2

    warped, record, weight_record, weight_warp = libwarp.forward_record(
        ref.contiguous(), grid.contiguous(), depth.contiguous())
    mask = weight_warp != 0
    residual = image * mask - warped

    grad = libwarp.backward_record(
        ref.contiguous(), grid.contiguous(), depth.contiguous(),
        record.contiguous(), weight_warp.contiguous(), residual.contiguous())

    writer.add_images('record/image', image, 0)
    writer.add_images('record/ref', ref, 0)
    writer.add_images('record/depth', depth, 0)
    writer.add_images('record/residual', residual, 0)
    writer.add_images('record/grid_u', grid[:, 0:1, ...] * 0.5 + 0.5, 0)
    writer.add_images('record/grid_v', grid[:, 1:2, ...] * 0.5 + 0.5, 0)
    writer.add_images('record/warped', warped, 0)
    writer.add_images('record/record', record, 0)
    writer.add_images('record/weight_record', weight_record, 0)
    writer.add_images('record/weight_warp', weight_warp, 0)
    writer.add_images('record/grad_u', grad[:, 0:1, ...].abs(), 0)
    writer.add_images('record/grad_v', grad[:, 1:2, ...].abs(), 0)
    writer.add_images('record/mask', mask, 0)



def test_warp(writer):
    image = torch.ones([1, 1, 100, 200], dtype=torch.float32, device='cuda:0')
    image[:, :, 20:80, 50:150] = 0.5
    ref = torch.ones([1, 1, 100, 200], dtype=torch.float32, device='cuda:0')
    ref[:, :, 30:90, 70:170] = 0.5

    depth = torch.ones([1, 1, 100, 200], dtype=torch.float32, device='cuda:0') * 0.4
    depth[:, :, 20:80, 50:150] = 0.8

    grid_x = torch.linspace(-1.0, 1.0, 200)
    grid_y = torch.linspace(-1.0, 1.0, 100)

    grid_y, grid_x = torch.meshgrid([grid_y, grid_x])
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([1, 1, 1, 1])
    grid = grid.to('cuda:0')
    grid[:, :, 20:80, 50:150] += 0.2

    warped = libwarp.forward(ref.contiguous(), grid.contiguous())
    residual = image - warped
    grad = libwarp.backward(ref.contiguous(), grid.contiguous(), residual.contiguous())

    writer.add_images('warp/image', image, 0)
    writer.add_images('warp/ref', ref, 0)
    writer.add_images('warp/depth', depth, 0)
    writer.add_images('warp/residual', residual, 0)
    writer.add_images('warp/grid_u', grid[:, 0:1, ...] * 0.5 + 0.5, 0)
    writer.add_images('warp/grid_v', grid[:, 1:2, ...] * 0.5 + 0.5, 0)
    writer.add_images('warp/warped', warped, 0)
    writer.add_images('warp/grad_u', grad[:, 0:1, ...].abs(), 0)
    writer.add_images('warp/grad_v', grad[:, 1:2, ...].abs(), 0)


if __name__ == '__main__':
    writer = SummaryWriter()
    test_warp(writer)
    test_record(writer)
    writer.close()
    # test_warp_direct()

