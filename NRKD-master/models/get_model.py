from models.cifar import cifar_model_dict
from models.others import others_model_dict


def get_model(name, dataset, num_classes):
    if dataset in ["cifar10", "cifar100", "stl10", "tiny_imagenet", "mnist"]:
        model = cifar_model_dict[name](num_classes=num_classes)
    else:
        model = others_model_dict[name](num_classes=num_classes)
    return model
