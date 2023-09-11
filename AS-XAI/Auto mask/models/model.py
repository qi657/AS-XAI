import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from ..models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from ..models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from ..util.receptive_field import compute_proto_layer_rf_info_v2

from ..core.sobel_loss import *

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class TESNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(TESNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function #log

        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity  每个元素的one-hot编码
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features_0 = features

        # resnet50
        # resnet = models.resnet50(pretrained=True)
        # self.features = nn.Sequential(*list(resnet.children())[:-1])


        # vgg19
        vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.children())[:-1])
        self.features.add_module('global average', nn.AvgPool2d(7))

        # densenet121
        # densenet = models.densenet161(pretrained=True)
        # self.features = nn.Sequential(*list(densenet.children()))[:-1]
        # self.features.add_module('1', self.features[0])
        # self.features._modules.pop('0')
        # self.features.add_module('global average', nn.AvgPool2d(1))


        features_name = str(self.features_0).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features_0.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features_0.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
                )

        # self.avg_layer = nn.AdaptiveAvgPool2d((1,1))

        # 初始化原型vector符合正态分布
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)


        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)

        if init_weights:
            self._initialize_weights()


    def conv_features(self, x):

        x = self.features_0(x)
        x = self.add_on_layers(x)

        return x

    def _cosine_convolution(self, x):
        # compactness loss (类内接近)
        x = F.normalize(x,p=2,dim=1)  # L2 norm  对应公式中的||P||
        # x = x + torch.randn_like(x) * 0.001  # P’
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)  # 公式中的b_j
        now_prototype_vectors = now_prototype_vectors + torch.randn_like(now_prototype_vectors) * 0.001  # b_j+正态偏移
        distances = F.conv2d(input=x, weight=now_prototype_vectors)  # 计算每个特征图P与b_j的距离，即b_j/||P||
        distances = -distances

        return distances


    def _project2basis(self,x):   # 投影度量
        # separation loss (类间分离)  相似性分数的激活图
        # x = x + torch.randn_like(x) * 0.001  # P’
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        now_prototype_vectors = now_prototype_vectors + torch.randn_like(now_prototype_vectors) * 0.001 # # b_j+正态偏移
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        return distances

    def prototype_distances(self, x):
        # rgb2hsv颜色空间转换
        # x = torch.stack([color_space_transform(img) for img in x]).cuda()
        # high-level 补丁分组
        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)  # 类内聚合
        project_distances = self._project2basis(conv_features)  # 类间分离

        return project_distances,cosine_distances


    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def global_min_pooling(self,distances):

        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)

        return min_distances

    def global_max_pooling(self,distances):

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances


    def forward(self, x):
        x = x.cuda()
        conv_features = self.features(x)
        x_feature = conv_features[:, :, 0, 0]
        project_distances,cosine_distances = self.prototype_distances(x)
        cosine_min_distances = self.global_min_pooling(cosine_distances)

        project_max_distances = self.global_max_pooling(project_distances)
        prototype_activations = project_max_distances
        logits = self.last_layer(prototype_activations)
        return logits, cosine_min_distances, x_feature

    def push_forward(self, x):

        conv_features = self.features(x)
        x_feature = conv_features[:, :, 0, 0]

        conv_output = self.conv_features(x) #[batchsize,128,14,14]

        distances = self._project2basis(conv_output)
        distances = - distances
        return conv_output, distances, x_feature

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength  # 初始值
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)  # 分类器的权重矩阵G



def construct_TesNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(700, 64, 1, 1), num_classes=120,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,  # 224
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return TESNet(features=features,
                  img_size=img_size,
                  prototype_shape=prototype_shape,
                  proto_layer_rf_info=proto_layer_rf_info,
                  num_classes=num_classes,
                  init_weights=True,
                  prototype_activation_function=prototype_activation_function,
                  add_on_layers_type=add_on_layers_type)

