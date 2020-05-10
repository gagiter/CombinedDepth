
import torch
import libwarp
from torch.utils.tensorboard import SummaryWriter


class WarpFuncion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample, depth):
        warped, weight, mask = libwarp.forward(
            image.contiguous(), sample.contiguous(), depth.contiguous())
        ctx.save_for_backward(image, sample, depth, weight)
        return warped

    @staticmethod
    def backward(ctx, grad_warp):
        image, sample, depth, weight = ctx.saved_tensors
        grad_image = grad_sample = grad_occlusion = None
        if ctx.needs_input_grad[1]:
            grad_sample = libwarp.backward(
                image.contiguous(), sample.contiguous(), depth.contiguous(),
                weight.contiguous(), grad_warp.contiguous())

        return grad_image, grad_sample, grad_occlusion


def test1():
    image = torch.ones([2, 1, 128, 256], dtype=torch.float32, device='cuda:0')
    image[:, :, 32:96, 96:160] = 0.5

    depth = torch.ones([2, 1, 128, 256], dtype=torch.float32, device='cuda:0') * 0.2
    depth[:, :, 32:96, 96:160] = 0.8

    grid_x = torch.linspace(-1.0, 1.0, 256)
    grid_y = torch.linspace(-1.0, 1.0, 128)

    grid_y, grid_x = torch.meshgrid([grid_y, grid_x])
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([2, 1, 1, 1])
    grid = grid.to('cuda:0')
    grid[:, :, 32:96, 96:160] += 0.1

    warp, weight, mask = WarpFuncion.apply(image, grid, depth)

    writer = SummaryWriter()
    writer.add_images('image/image', image, 0)
    writer.add_images('image/depth', depth, 0)
    writer.add_images('image/grid', grid[:, 0:1, ...] * 0.5 + 0.5, 0)
    writer.add_images('image/warp', warp, 0)
    writer.add_images('image/weight', weight, 0)
    writer.add_images('image/mask', mask, 0)
    writer.close()


def test2():

    image = torch.ones([1, 1, 128, 256], dtype=torch.float32, device='cuda:0')
    ref = torch.ones([1, 1, 128, 256], dtype=torch.float32, device='cuda:0')
    depth = torch.rand([1, 1, 128, 256], dtype=torch.float32, device='cuda:0')

    image[:, :, 30:80, 100:150] = 0.0
    ref[:, :, 50:100, 120:170] = 0.0

    grid_x = torch.linspace(-1.0, 1.0, 256)
    grid_y = torch.linspace(-1.0, 1.0, 128)

    grid_y, grid_x = torch.meshgrid([grid_y, grid_x])
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([1, 1, 1, 1])
    grid = grid.to('cuda:0')

    writer = SummaryWriter()
    move = torch.zeros([1, 2, 1, 1], dtype=torch.float32, device='cuda:0')
    move.requires_grad_()
    optimizer = torch.optim.SGD([move], 0.01)

    for i in range(200):
        sample = grid + move
        warp, weight, mask = WarpFuncion.apply(image, sample, depth)
        residual_warp = (warp - mask * ref).abs()
        loss = residual_warp.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item(), move.grad.data[0, 0, 0, 0].item(), move.grad.data[0, 1, 0, 0].item())
        # print('move', move[0, :, 0, 0])
        writer.add_images('image/image', image, global_step=i)
        writer.add_images('image/ref', ref, global_step=i)
        writer.add_images('image/warp', warp, global_step=i)
        writer.add_images('image/weight', weight, global_step=i)
        writer.add_images('image/mask', mask, global_step=i)
        writer.add_images('image/residual_warp', residual_warp, global_step=i)
        writer.add_scalar('loss', loss, global_step=i)
        writer.add_scalar('move_x', move[0, 0, 0, 0], global_step=i)
        writer.add_scalar('move_y', move[0, 1, 0, 0], global_step=i)

    writer.close()


if __name__ == '__main__':
    # test1()
    test2()

