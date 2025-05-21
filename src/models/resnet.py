
import torch.nn as nn

def count_num_trainable_params(model):
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params before training: {params_before / 1_000_000:.2f}M")

def decide_model_params(model_name):
    model_params = {
        10: {"block": BasicBlock, "layers": [1, 1, 1, 1]},
        18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
        34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
        50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
        101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
        152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
    }

    if model_name == "ResNet10":
        return model_params[10]
    elif model_name == "ResNet18":
        return model_params[18]
    elif model_name == "ResNet34":
        return model_params[34]

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

def norm2d(group_norm_num_groups, planes, track_running_stats = True):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        group_nums = planes // group_norm_num_groups
        return nn.GroupNorm(group_nums, planes)
    else:
        return nn.BatchNorm2d(planes,track_running_stats = track_running_stats)

class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        #self.register_buffer("rate",torch.tensor(rate))

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output

# Basic Block for ResNet18, 34
class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """
    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats = True,
        rate = 1,
        need_scaler = False,
        global_rate = 1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(round(out_planes)), stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats = track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(int(round(out_planes)), int(round(out_planes)))
        self.bn2 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats = track_running_stats)

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

# Bottlenect Block for ResNet50, 101, 152
class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats=True,
        rate=1,
        need_scaler=False,
        rank_factor=1,
        square=False
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=int(round(out_planes)), kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats=track_running_stats)

        self.conv2 = nn.Conv2d(
            in_channels=int(round(out_planes)),
            out_channels=int(round(out_planes)),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=int(round(out_planes)), track_running_stats=track_running_stats)

        self.conv3 = nn.Conv2d(
            in_channels=int(round(out_planes)),
            out_channels=int(round(out_planes * 4)),
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=int(round(out_planes * 4)),
                          track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBase(nn.Module):    
    def _weight_initialization(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_block(self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None, track_running_stats=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion,track_running_stats=track_running_stats),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats = track_running_stats,

            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                    track_running_stats=track_running_stats
                )
            )
        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNetBase, self).train(mode)

class ResNet(ResNetBase):
    def __init__(
        self,
        model_name,
        dataset,
        num_classes,
        scaler_rate = 1,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection=False,
        save_activations=False,
        pruning = False,
        need_scaler = False,
        global_rate = 1,
        num_channel = 3,
    ):
        super(ResNet, self).__init__()
        self.model_name = model_name                # ResNet18, ResNet34, ResNet50, ...
        self.dataset = dataset                      # CIFAR10, CIFAR100
        self.num_classes = num_classes
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        self.need_scaler = need_scaler

        track_stat = not self.freeze_bn
        model_params = decide_model_params(self.model_name)
        block_fn = model_params["block"]
        block_nums = model_params["layers"]

        # define layers.
        self.inplanes = int(round(64 * scaler_rate))
        self.conv1 = nn.Conv2d(num_channel, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.need_scaler:
            self.scaler = Scaler(scaler_rate / global_rate)
        self.bn1 = norm2d(group_norm_num_groups, planes=self.inplanes, track_running_stats = track_stat)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_block(
            block_fn=block_fn,
            planes=64,
            stride=1,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate = scaler_rate,
            global_rate=global_rate
        )
        self.layer2 = self.make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate=scaler_rate,
            global_rate=global_rate
        )
        self.layer3 = self.make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate=scaler_rate,
            global_rate=global_rate
        )
        self.layer4 = self.make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_stat,
            scaler_rate=scaler_rate,
            global_rate=global_rate
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = projection

        if self.projection:
            self.projection_layer = nn.Sequential(
                nn.Linear(int(round(512 * block_fn.expansion * scaler_rate)), int(round(512 * block_fn.expansion * scaler_rate))),
                nn.ReLU(),
                nn.Linear(int(round(512 * block_fn.expansion* scaler_rate)), 256)
            )
            self.classifier = nn.Linear(
                in_features=256,
                out_features=self.num_classes,
            )
        else:
            self.classifier = nn.Linear(
                in_features=int(round(512 * block_fn.expansion * scaler_rate)), out_features=self.num_classes
            )
        self.save_activations = save_activations
        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward(self, x,start_layer_idx = 0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            if self.need_scaler:
                x = self.scaler(x)
            x = self.bn1(x)
            x = self.relu(x)

            #x = self.maxpool(x)
            x = self.layer1(x)
            activation1 = x
            x = self.layer2(x)
            activation2 = x
            x = self.layer3(x)
            activation3 = x
            x = self.layer4(x)
            activation4 = x
            x = self.avgpool(x)
            feature = x.view(x.size(0), -1)
            if self.projection:
                feature = self.projection_layer(feature)

            if self.save_activations:
                self.activations = [activation1, activation2, activation3,activation4]
        else:
            feature = x
        x = self.classifier(feature)
        return x

    def make_block(self,block_fn,planes,block_num,stride = 1,
                    group_norm_num_groups=None,
                    track_running_stats=True,
                    scaler_rate = 1,
                    global_rate = 1):
        downsample = None
        if stride != 1 or self.inplanes != int(round(planes * block_fn.expansion * scaler_rate)):
            if self.need_scaler:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        int(round(planes * block_fn.expansion * scaler_rate)) ,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    Scaler(scaler_rate / global_rate),
                    norm2d(group_norm_num_groups, planes=int(round(planes * block_fn.expansion * scaler_rate)),
                           track_running_stats=track_running_stats),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        int(round(planes * block_fn.expansion * scaler_rate)) ,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    norm2d(group_norm_num_groups, planes=int(round(planes * block_fn.expansion* scaler_rate)) ,
                           track_running_stats=track_running_stats),
                )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes= planes * scaler_rate,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats=track_running_stats,
                rate = scaler_rate,
                need_scaler = self.need_scaler,
                global_rate = global_rate
            )
        )
        self.inplanes = int(round(planes * block_fn.expansion* scaler_rate))

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes= planes * scaler_rate,
                    group_norm_num_groups=group_norm_num_groups,
                    track_running_stats=track_running_stats,
                    rate = scaler_rate,
                    need_scaler=self.need_scaler,
                    global_rate=global_rate
                )
            )
        return nn.Sequential(*layers)

def fullrank_resnet_loader(args):
    model_name = args.model_name
    dataset = args.dataset
    num_classes = args.num_classes
    freeze_bn = True

    if args.dataset == "FashionMNIST":
        num_channel = 1
    else:
        num_channel = 3

    model = ResNet(
        model_name=model_name,
        dataset=dataset,
        num_classes=num_classes,
        scaler_rate=1,
        freeze_bn=freeze_bn,
        num_channel=num_channel
    )

    return model