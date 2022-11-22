import torch
import torch.nn as nn
import torch.nn.functional as F
import layers


class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder = Encoder()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

    def forward(self, input):
        logits = self.encoder(input)
        logits = F.relu(self.bn1(self.fc1(logits)))
        logits = self.drop1(logits)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = self.drop2(logits)
        logits = self.fc3(logits)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs, logits

    
class Encoder(nn.Module):
    def __init__(self, input_channels=1, activation='relu'):
        super(Encoder, self).__init__()

        self.view_resnet = resnet22(input_channels, activation)
        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

    def forward(self, x):
        h = self.all_views_gaussian_noise_layer.single_add_gaussian_noise(x)
        result = self.view_resnet(h)
        h = self.all_views_avg_pool.single_avg_pool(result)
        return h

    def load_state_from_shared_weights(self, state_dict, view):
        view_angle = view.lower().split("-")[-1]
        view_key = view.lower().replace("-", "")
        self.view_resnet.load_state_dict(
            filter_strip_prefix(state_dict, "four_view_resnet.{}.".format(view_angle))
        )

        
class Discriminator(nn.Module):
    def __init__(self, dim_in=256):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_in, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        #noise = noise.to(device)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


class ViewResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    def __init__(self,
                 input_channels, activation, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ViewResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(current_num_filters // growth_factor * block_fn.expansion)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.activation(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


def resnet22(input_channels, activation):
    return ViewResNetV2(
        input_channels=input_channels,
        activation=activation,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        block_fn=layers.BasicBlockV2,
        first_layer_padding=0,
        first_pool_size=3,
        first_pool_stride=2,
        first_pool_padding=0,
        growth_factor=2,
    )


def filter_strip_prefix(weights_dict, prefix):
    return {
        k.replace(prefix, ""): v
        for k, v in weights_dict.items()
        if k.startswith(prefix)
    }
