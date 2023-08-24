from .cifar100 import get_cifar100_dataloaders
from .imagenet import get_imagenet_dataloaders
from .tiny_imagenet import get_tinyimagenet_dataloaders
from .cifar10 import get_cifar10_dataloaders


NUM_WORKERS = 8
def get_dataloader(args):
    if args.dataset == "cifar100":
        train_loader, val_loader, num_data = get_cifar100_dataloaders(
            batch_size=args.batch_size,
            val_batch_size=args.batch_size // 2,
            num_workers=NUM_WORKERS,
        )
        num_classes = 100
    elif args.dataset == "imagenet":
        train_loader, val_loader, num_data = get_imagenet_dataloaders(
            batch_size=args.batch_size,
            val_batch_size=args.batch_size // 2,
            num_workers=NUM_WORKERS,
        )
        num_classes = 1000

    elif args.dataset == "tiny_imagenet":
        train_loader, val_loader, num_data = get_tinyimagenet_dataloaders(
            batch_size=args.batch_size,
            val_batch_size=args.batch_size // 2,
            num_workers=NUM_WORKERS,
        )
        num_classes = 200
    elif args.dataset == "cifar10":
        train_loader, val_loader, num_data = get_cifar10_dataloaders(
            batch_size=args.batch_size,
            val_batch_size=args.batch_size // 2,
            num_workers=NUM_WORKERS
        )
        num_classes = 10
    else:
        raise NotImplementedError(args.dataset)

    return train_loader, val_loader, num_data, num_classes
