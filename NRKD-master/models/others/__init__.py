from .resnetv2 import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv2 import mobilenet_v2
from .shufflenetv2 import shufflenet_v2_x1_0

others_model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "mobilenetv2": mobilenet_v2,
    "shufflenetv2": shufflenet_v2_x1_0
}