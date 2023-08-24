from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet32x4, resnet8x4
from .resnetv2 import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .mobilenetv2 import mobile_half
from .shufflenetv1 import ShuffleNetV1
from .shufflenetv2 import ShuffleV2

cifar_resnetv1_size_dict = {
    "resnet20": [[16, 32, 32], [16, 32, 32], [32, 16, 16], [64, 8, 8], [64, ]],
    "resnet32": [[16, 32, 32], [16, 32, 32], [32, 16, 16], [64, 8, 8], [64, ]],
    "resnet56": [[16, 32, 32], [16, 32, 32], [32, 16, 16], [64, 8, 8], [64, ]],
    "resnet32x4": [[32, 32, 32], [64, 32, 32], [128, 16, 16], [256, 8, 8], [256, ]],
    "resnet8x4": [[32, 32, 32], [64, 32, 32], [128, 16, 16], [156, 8, 8], [256, ]],
    "resnet110": [[16, 32, 32], [16, 32, 32], [32, 16, 16], [64, 8, 8], [64, ]],
}

cifar_resnetv2_size_dict = {
    "resnet18": [[64, 32, 32], [64, 32, 32], [128, 16, 16], [256, 8, 8], [512, 4, 4], [512]],
    "resnet34": [[64, 32, 32], [64, 32, 32], [128, 16, 16], [256, 8, 8], [512, 4, 4], [512]],
    "resnet50": [[64, 32, 32], [256, 32, 32], [512, 16, 16], [1024, 8, 8], [2048, 4, 4], [2048]],
    "resnet101": [[64, 32, 32], [256, 32, 32], [512, 16, 16], [1024, 8, 8], [2048, 4, 4], [2048]],
}

cifar_wrn_size_dict = {
    "wrn_16_2": [[16, 32, 32], [32, 32, 32], [64, 16, 16], [128, 8, 8], [128]],
    "wrn_40_2": [[16, 32, 32], [32, 32, 32], [64, 16, 16], [128, 8, 8], [128]],
    "wrn_16_1": [[16, 32, 32], [16, 32, 32], [32, 16, 16], [64, 8, 8], [64]],
    "wrn_40_1": [[16, 32, 32], [16, 32, 32], [32, 16, 16], [64, 8, 8], [64]],
}

cifar_vgg_size_dict = {
    "vgg8": [[64, 32, 32], [128, 16, 16], [256, 8, 8], [512, 4, 4], [512, 4, 4], [512]],
    "vgg13": [[64, 32, 32], [128, 16, 16], [256, 8, 8], [512, 4, 4], [512, 4, 4], [512]],
}

cifar_shufflenet_size_dict = {
    "shufflenetv1": [[24, 32, 32], [240, 16, 16], [480, 8, 8], [960, 4, 4], [960]],
    "shufflenetv2": [[24, 32, 32], [116, 16, 16], [232, 8, 8], [464, 4, 4], [1024]],
}

cifar_mobilenetv2_size_dict = {
    "mobilenetv2": [[16, 16, 16], [12, 16, 16], [16, 8, 8], [48, 4, 4], [160, 2, 2], [1280]]
}

cifar_model_dict = {
    "resnet8": resnet8,
    "resnet14": resnet14,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet32x4": resnet32x4,
    "resnet8x4": resnet8x4,
    "vgg8": vgg8_bn,
    "vgg13": vgg13_bn,
    "wrn_16_2": wrn_16_2,
    "wrn_40_2": wrn_40_2,
    "wrn_16_1": wrn_16_1,
    "wrn_40_1": wrn_40_1,
    "shufflenetv1": ShuffleNetV1,
    "shufflenetv2": ShuffleV2,
    "mobilenetv2": mobile_half,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}