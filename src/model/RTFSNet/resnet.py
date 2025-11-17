import torch
from torch import nn
from typing import Optional

class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu2(out + residual)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        layers: list[int],
        gamma_zero: bool = False,
        avg_pool_downsample: bool = False
    ):
        super().__init__()
        self.inplanes = 64
        self.gamma_zero = gamma_zero
        self.avg_pool_downsample = avg_pool_downsample

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        if stride != 1 or self.inplanes != planes * ResnetBlock.expansion:
            if self.avg_pool_downsample:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False
                    ),
                    nn.Conv2d(
                        in_channels=self.inplanes,
                        out_channels=planes * ResnetBlock.expansion,
                        kernel_size=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(planes * ResnetBlock.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.inplanes,
                        out_channels=planes * ResnetBlock.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm2d(planes * ResnetBlock.expansion),
                )
        else:
            downsample = None

        layers = [ResnetBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * ResnetBlock.expansion
        layers += [ResnetBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
