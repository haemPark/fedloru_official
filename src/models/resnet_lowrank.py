
'''
    This is for FedHM
    The code is from authors of FedHM by request
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class FullRankBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        rank_factor=None,
        track_running_stats=True,
        square=True
    ):
        super(FullRankBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes,track_running_stats = track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes,track_running_stats = track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class FullRankBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        rank_factor=None,
        track_running_stats=False,
        square = False
    ):  # we add a dummy parameter here to make the APIs more adaptable
        super(FullRankBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width,track_running_stats=track_running_stats)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width,track_running_stats=track_running_stats)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LowRankBasicBlockConv1x1(nn.Module):
    expansion = 1
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        rank_factor=4,
        track_running_stats=True,
        square=False
    ):
        super(LowRankBasicBlockConv1x1, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.square = square

        if square:
            dim1, dim2 = planes , inplanes * 3 * 3
        else:
            dim1, dim2 = planes * 3, inplanes * 3

        self.out_channels , self.in_channels, self.stride, self.dilation,self.padding, self.groups = \
            planes, inplanes, stride, dilation, 1, groups

        # rank should be at least equal with kernel size
        self.rank = max(int(round(planes / rank_factor)), 1 )

        self.conv1_u = nn.Parameter(torch.zeros(self.rank, dim2))
            # conv1x3(inplanes, self.rank,
            #                    stride = (1,stride),
            #                    padding = (0 , dilation),
            #                    dilation = (1 , dilation))


        #self.bn1_u = norm_layer(self.rank,track_running_stats=track_running_stats)
        self.conv1_v = nn.Parameter(torch.zeros(dim1, self.rank))
            # conv3x1(self.rank, planes,
            #                    stride = (stride, 1),
            #                    padding=(dilation, 0),
            #                    dilation=(dilation , 1))
        #self.bn1_v = norm_layer(planes)
        # if self.stride > 1:
        #     self.downsample2 = nn.Sequential(
        #         conv1x1(self.in_channels, planes, stride),
        #         norm_layer(planes)
        #     )
        # else:
        #     self.downsample2 = None
        self.bn1 = norm_layer(planes,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        if square:
            dim1, dim2 = planes, planes * 3 * 3
        else:
            dim1, dim2 = planes * 3, planes * 3

        self.conv2_u = nn.Parameter(torch.zeros(self.rank, dim2))
            # conv1x3(planes, self.rank,
            #                    stride=(1, 1),
            #                    padding=(0, dilation),
            #                    dilation=(1, dilation))
        #self.bn2_u = norm_layer(self.rank,track_running_stats=track_running_stats)
        self.conv2_v = nn.Parameter(torch.zeros(dim1, self.rank))
            # conv3x1(self.rank, planes,
            #                    stride=(1, 1),
            #                    padding=(dilation, 0),
            #                    dilation=(dilation, 1))
        #self.bn2_v = norm_layer(planes)
        self.bn2 = norm_layer(planes,track_running_stats=track_running_stats)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.square:
            out = F.conv2d(x,
                           self.conv1_u.reshape(self.rank, self.in_channels, 3 ,3 ),
                           None,
                           stride = self.stride,
                           padding = 1,
                           dilation = (1, self.dilation),
                           groups=self.groups).contiguous()
            out = self.bn1_u(out)
            out = F.conv2d(out,
                           self.conv1_v.reshape(self.out_channels, self.rank, 1, 1),
                           None,
                           stride=1,
                           padding=0,
                           dilation=(self.dilation, 1),
                           groups=self.groups).contiguous()
        else:
            out = F.conv2d(x,
                           self.conv1_u.T.reshape(self.in_channels, 3, 1, self.rank).permute(3, 0, 2, 1),
                           None,
                           stride=(1, self.stride),
                           padding=(0, self.padding),
                           dilation=(1, self.dilation),
                           groups=self.groups).contiguous()
            #out = self.bn1_u(out)
            out = F.conv2d(out,
                           self.conv1_v.reshape(self.out_channels, 3, self.rank, 1).permute(0, 2, 1, 3),
                           None,
                           stride=(self.stride, 1),
                           padding=(self.padding, 0),
                           dilation=(self.dilation, 1),
                           groups=self.groups).contiguous()

        out = self.bn1(out)
        out = self.relu(out)

        if self.square:
            out = F.conv2d(out,
                           self.conv2_u.reshape(self.rank, self.out_channels, 3, 3),
                           None,
                           stride=1,
                           padding=1,
                           dilation=(1, self.dilation),
                           groups=self.groups).contiguous()
            out = self.bn2_u(out)
            out = F.conv2d(out,
                           self.conv2_v.reshape(self.out_channels, self.rank, 1, 1),
                           None,
                           stride=1,
                           padding=0,
                           dilation=(self.dilation, 1),
                           groups=self.groups).contiguous()
        else:
            out = F.conv2d(out,
                           self.conv2_u.T.reshape(self.out_channels, 3, 1, self.rank).permute(3, 0, 2, 1),
                           None,
                           stride=1,
                           padding=(0, self.padding),
                           dilation=(1, self.dilation),
                           groups=self.groups).contiguous()
            #out = self.bn2_u(out)
            out = F.conv2d(out,
                           self.conv2_v.reshape(self.out_channels, 3, self.rank, 1).permute(0, 2, 1, 3),
                           None,
                           stride=1,
                           padding=(self.padding, 0),
                           dilation=(self.dilation, 1),
                           groups=self.groups).contiguous()

        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class LowRankBottleneckConv1x1(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        rank_factor=4,
        track_running_stats=False,
        square=False
    ):
        super(LowRankBottleneckConv1x1, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1_u = conv3x3(inplanes, int(planes / rank_factor), stride)
        self.square = square

        dim1, dim2 = width, inplanes

        self.out_channels, self.in_channels, self.stride, self.dilation, self.padding, self.groups = \
            width, inplanes, stride, dilation, 1, groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.rank = max(int(round(width / rank_factor)), 1)
        self.conv1_u = nn.Parameter(torch.zeros(self.rank, dim2))
        self.conv1_v = nn.Parameter(torch.zeros(dim1, self.rank))
        #self.conv1_u = conv1x1(inplanes, int(width / rank_factor))
        # self.conv1_v = conv1x1(int(width / rank_factor), width)
        self.bn1 = norm_layer(width,track_running_stats=track_running_stats)

        if square:
            dim1, dim2 = width, width * 3 * 3
        else:
            dim1, dim2 = width * 3, width * 3
        self.conv2_u = nn.Parameter(torch.zeros(self.rank, dim2))
        self.conv2_v = nn.Parameter(torch.zeros(dim1, self.rank))

        #self.conv2_u = conv3x3(width, int(width / rank_factor), stride, groups, dilation)
        #self.conv2_v = conv1x1(int(width / rank_factor), width)
        self.bn2 = norm_layer(width,track_running_stats=track_running_stats)

        dim1, dim2 = planes * self.expansion, width

        self.conv3_u = nn.Parameter(torch.zeros(self.rank, dim2))
        self.conv3_v = nn.Parameter(torch.zeros(dim1, self.rank))
        # self.conv3_u = conv1x1(width, int(width / rank_factor))
        # self.conv3_v = conv1x1(int(width / rank_factor), planes * self.expansion)
        # self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion,track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.conv2d(x,
                       self.conv1_u.T.reshape(self.in_channels, 1, 1, self.rank).permute(3, 0, 2, 1),
                       None,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1).contiguous()

        out = F.conv2d(out,
                       self.conv1_v.reshape(self.out_channels, 1, self.rank, 1).permute(0, 2, 1, 3),
                       None,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1).contiguous()

        out = self.bn1(out)
        out = self.relu(out)

        out = F.conv2d(out,
                       self.conv2_u.T.reshape(self.out_channels, 3, 1, self.rank).permute(3, 0, 2, 1),
                       None,
                       stride=(1,self.stride),
                       padding=(0, self.padding),
                       dilation=(1, self.dilation),
                       groups=self.groups).contiguous()
        # out = self.bn2_u(out)
        out = F.conv2d(out,
                       self.conv2_v.reshape(self.out_channels, 3, self.rank, 1).permute(0, 2, 1, 3),
                       None,
                       stride=(self.stride,1),
                       padding=(self.padding, 0),
                       dilation=(self.dilation, 1),
                       groups=self.groups).contiguous()
        out = self.bn2(out)
        out = self.relu(out)

        out = F.conv2d(out,
                       self.conv3_u.T.reshape(self.out_channels, 1, 1, self.rank).permute(3, 0, 2, 1),
                       None,
                       1,
                       padding=0,
                       dilation=1,
                       groups=1).contiguous()
        # out = self.bn1_u(out)
        out = F.conv2d(out,
                       self.conv3_v.reshape(self.out_channels * self.expansion, 1, self.rank, 1).permute(0, 2, 1, 3),
                       None,
                       stride=1,
                       padding=0,
                       groups=1).contiguous()

        # out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HybridResNet(nn.Module):
    def __init__(
        self,
        lowrank_block,
        fullrank_block,
        rank_factor,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        track_running_stats=True,
        scaler_rate=1,
        square=False,
        num_channel = 3,
    ):
        # (lowrank_block, fullrank_block, rank_factor, layers, **kwargs)
        super(HybridResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.rank_factor = rank_factor

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if rank_factor > 256:
            blocks = [[lowrank_block]*layers[0], [lowrank_block]*layers[1],
                        [lowrank_block]*layers[2], [lowrank_block]*layers[3]]
        elif rank_factor > 128:
            blocks = [[fullrank_block] * layers[0], [lowrank_block] * layers[1],
                      [lowrank_block] * layers[2], [lowrank_block] * layers[3]]
        else:
            if layers[0] != layers[1]:
                blocks = [[fullrank_block] * layers[0],
                          [fullrank_block] * layers[1],
                          [lowrank_block] * layers[2], [lowrank_block] * layers[3]]
            else:
                blocks = [[fullrank_block,lowrank_block], [lowrank_block,lowrank_block],
                          [lowrank_block,lowrank_block], [lowrank_block,lowrank_block]]
        planes = [ int(round( p * scaler_rate)) for p in [64, 128, 256, 512]]

        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(num_channel, self.inplanes, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(blocks[0], planes[0], layers[0],rank_factor=rank_factor,
                                       track_running_stats=track_running_stats,square=square)
        self.layer2 = self._make_layer(blocks[1], planes[1], layers[1], stride=2,rank_factor=rank_factor,
                                       dilate=replace_stride_with_dilation[0],
                                       track_running_stats=track_running_stats,square=square )
        self.layer3 = self._make_layer(blocks[2], planes[2], layers[2], stride=2,rank_factor=rank_factor,
                                       dilate=replace_stride_with_dilation[1],
                                       track_running_stats=track_running_stats,square=square)
        self.layer4 = self._make_layer(blocks[3], planes[3], layers[3], rank_factor=rank_factor, stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       track_running_stats=track_running_stats,square=square)
        # ================================================================================================================

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * fullrank_block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, rank_factor=None, dilate=False,
                    track_running_stats = True, square = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block[0].expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block[0].expansion, stride),
                norm_layer(planes * block[0].expansion,track_running_stats=track_running_stats),
            )

           
        layers = []
        layers.append(block[0](self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor,
                            track_running_stats=track_running_stats,square=square))
        self.inplanes = planes * block[0].expansion

        for i in range(1, blocks):
            layers.append(block[i](self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, rank_factor=rank_factor,
                                track_running_stats = track_running_stats,
                                square = square ))

        return nn.Sequential(*layers)

    def _make_layer_dual_blocks(self, fr_block, lr_block, planes, blocks, stride=1, rank_factor=None, dilate=False):
        """
        Trial implementation: just for `Layer3` in resnet50
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * blocks.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * fr_block.expansion, stride),
                norm_layer(planes * fr_block.expansion),
            )
        layers = []
        layers.append(fr_block(self.inplanes, planes, stride, downsample, self.groups,
                               self.base_width, previous_dilation, norm_layer, rank_factor=rank_factor))
        self.inplanes = planes * fr_block.expansion
        for block_index in range(1, blocks):
            if block_index <= 2:
                layers.append(fr_block(self.inplanes, planes, groups=self.groups,
                                       base_width=self.base_width, dilation=self.dilation,
                                       norm_layer=norm_layer, rank_factor=rank_factor))
            else:
                layers.append(lr_block(self.inplanes, planes, groups=self.groups,
                                       base_width=self.base_width, dilation=self.dilation,
                                       norm_layer=norm_layer, rank_factor=rank_factor))
        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = self.fc(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


lowrank_model_params = {
    10: {"block": [LowRankBasicBlockConv1x1,FullRankBasicBlock], "layers": [1, 1, 1, 1]},
    18: {"block": [LowRankBasicBlockConv1x1,FullRankBasicBlock], "layers": [2, 2, 2, 2]},
    34: {"block": [LowRankBasicBlockConv1x1,FullRankBasicBlock], "layers": [3, 4, 6, 3]},
    50: {"block": [LowRankBottleneckConv1x1,FullRankBottleneck], "layers": [3, 4, 6, 3]},
    101: {"block": [LowRankBottleneckConv1x1,FullRankBottleneck], "layers": [3, 4, 23, 3]},
    152: {"block": [LowRankBottleneckConv1x1,FullRankBottleneck], "layers": [3, 8, 36, 3]},
}

def lowrank_resnet_loader(args, rank_factor):
    model_name = args.model_name
    num_classes = args.num_classes
    freeze_bn=True

    if model_name == 'ResNet10':
        resnet_size = 10
    elif model_name == 'ResNet18':
        resnet_size = 18
    elif model_name == 'ResNet34':
        resnet_size = 34
    elif model_name == 'ResNet50':
        resnet_size = 50
    elif model_name == 'ResNet101':
        resnet_size = 101
    elif model_name == 'ResNet152':
        resnet_size = 152

    if args.dataset == "FashionMNIST":
        num_channel = 1
    else:
        num_channel = 3

    block = lowrank_model_params[resnet_size]["block"]
    layers = lowrank_model_params[resnet_size]["layers"]

    model = HybridResNet(
        block[0], block[1],
        rank_factor=rank_factor,
        layers=layers,
        num_classes=num_classes,
        track_running_stats=not freeze_bn,
        num_channel = num_channel
    )

    return model
