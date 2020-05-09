
import torch
import libwarp
from torch.utils.tensorboard import SummaryWriter


class WarpFuncion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample, occlusion=None):
        if occlusion is None:
            warped = libwarp.forward(image.contiguous(), sample.contiguous())
            ctx.save_for_backward(image, sample, occlusion)
        else:
            warped = libwarp.forward_with_occlusion(
                image.contiguous(), sample.contiguous(), occlusion.contiguous())
            ctx.save_for_backward(image, sample, occlusion)
        # print("warped", warped)
        return warped

    @staticmethod
    def backward(ctx, grad_output):
        # print("grad_output", grad_output)
        image, sample, occlusion = ctx.saved_tensors
        grad_image = grad_sample = grad_occlusion = None
        if ctx.needs_input_grad[1]:
                if occlusion is None:
                    grad_sample = libwarp.backward(
                        image.contiguous(), sample.contiguous(), grad_output.contiguous())
                else:
                    grad_sample = libwarp.backward_with_occlusion(
                        image.contiguous(), sample.contiguous(), occlusion.contiguous(), grad_output.contiguous())
        # print("grad_sample", grad_sample)
        return grad_image, grad_sample, grad_occlusion


def test1():
    image = torch.ones([2, 1, 128, 256], dtype=torch.float32, device='cuda:0')
    image[:, :, 32:96, 96:160] = 0.5

    depth = torch.ones([2, 1, 128, 256], dtype=torch.float32, device='cuda:0') * 0.5
    depth[:, :, 32:96, 96:160] = 1.0

    grid_x = torch.linspace(-1.0, 1.0, 256)
    grid_y = torch.linspace(-1.0, 1.0, 128)

    grid_y, grid_x = torch.meshgrid([grid_y, grid_x])
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([2, 1, 1, 1])
    grid = grid.to('cuda:0')
    grid[:, :, 32:96, 96:160] += 0.1

    warp = WarpFuncion.apply(image, grid, depth)
    warp2 = WarpFuncion.apply(image, grid)

    writer = SummaryWriter()
    writer.add_images('image/image', image, 0)
    writer.add_images('image/depth', depth, 0)
    writer.add_images('image/grid', grid[:, 0:1, ...] * 0.5 + 0.5, 0)
    writer.add_images('image/warp', warp, 0)
    writer.add_images('image/warp2', warp2, 0)
    writer.close()


def test2():

    image = torch.ones([1, 1, 128, 256], dtype=torch.float32, device='cuda:0')
    ref = torch.ones([1, 1, 128, 256], dtype=torch.float32, device='cuda:0')
    occlusion = torch.rand([1, 1, 128, 256], dtype=torch.float32, device='cuda:0')

    image[:, :, 30:80, 100:150] = 0.0
    ref[:, :, 50:100, 120:170] = 0.0
    # image[:, :, :, 1:5] = 0.0
    # ref[:, :, :, 2:6] = 0.0

    # print('image', image)
    # print('ref', ref)

    grid_x = torch.linspace(-1.0, 1.0, 256)
    grid_y = torch.linspace(-1.0, 1.0, 128)

    grid_y, grid_x = torch.meshgrid([grid_y, grid_x])
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([1, 1, 1, 1])
    grid = grid.to('cuda:0')

    # warp_0 = WarpFuncion.apply(image, grid, occlusion)
    # residual_ref = (image - ref).abs()
    # grid_v = torch.cat([grid * 0.5 + 0.5, torch.ones([2, 1, 128, 256], device='cuda:0')], dim=1)

    writer = SummaryWriter()
    # writer.add_images('image/image', image, 0)
    # writer.add_images('image/ref', ref, 0)
    # writer.add_images('image/warp_0', warp_0, 0)
    # writer.add_images('image/residual_ref', residual_ref, 0)
    # writer.add_images('image/grid_v', grid_v, 0)

    move = torch.zeros([1, 2, 1, 1], dtype=torch.float32, device='cuda:0')
    # move *= 0.01
    move.requires_grad_()
    optimizer = torch.optim.SGD([move], 0.01)

    for i in range(200):
        sample = grid + move
        warp = WarpFuncion.apply(image, sample, occlusion)
        residual_warp = (warp - ref).abs()
        loss = residual_warp.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item(), move.grad.data[0, 0, 0, 0].item(), move.grad.data[0, 1, 0, 0].item())
        # print('move', move[0, :, 0, 0])
        writer.add_images('image/image', image, global_step=i)
        writer.add_images('image/ref', ref, global_step=i)
        writer.add_images('image/warp', warp, global_step=i)
        writer.add_images('image/residual_warp', residual_warp, global_step=i)
        writer.add_scalar('loss', loss, global_step=i)
        writer.add_scalar('move_x', move[0, 0, 0, 0], global_step=i)
        writer.add_scalar('move_y', move[0, 1, 0, 0], global_step=i)

    writer.close()


if __name__ == '__main__':
    test1()
    test2()

