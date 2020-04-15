
import torch.nn


class Block(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, dilation, kernels):
        super(Block, self).__init__()
        self.branches = torch.nn.ModuleList()
        for kernel in kernels:
            padding = dilation * ((kernel - 1) // 2)
            branch = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, conv_channels, kernel_size=kernel, padding=padding,
                                padding_mode='zeros', dilation=dilation),
                torch.nn.BatchNorm2d(conv_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel, padding=padding,
                                padding_mode='zeros', dilation=dilation),
                torch.nn.BatchNorm2d(conv_channels),
                torch.nn.ReLU())
            self.branches.append(branch)

    def forward(self, x):
        y = None
        for branch in self.branches:
            xx = branch(x)
            y = xx if y is None else y + xx
        return y


class Aspp(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, dilations, kernels):
        super(Aspp, self).__init__()
        self.branches = torch.nn.ModuleList()
        for dilation in dilations:
            block = Block(in_channels, conv_channels, dilation, kernels)
            self.branches.append(block)

    def forward(self, x):
        y = None
        for branch in self.branches:
            xx = branch(x)
            y = xx if y is None else y + xx
        return y


class Dense(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Dense, self).__init__()
        self.out_channels = out_channels
        blocks = []
        # channels = [4, 16, 32, 64, 32, 16, 8, 4]
        channels = [4, 4, 8, 8, 16, 32, 32, 16, 8, 8, 8, 4]
        dilations = [1, 2, 4]
        kernels = [3, 5, 7]

        for out_channel in channels:
            blocks.append(Aspp(in_channels, out_channel, dilations, kernels))
            in_channels = out_channel

        blocks.append(torch.nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3,
                                      padding=1, padding_mode='zeros'))
        blocks.append(torch.nn.Sigmoid())
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)
