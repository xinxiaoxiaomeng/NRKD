from .utils import AverageMeter, accuracy
import torch

def test_cls(model, test_loader, criterion, args):

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()
    print("测试：")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            fea, logits = model(inputs, is_feat=True)

            loss = criterion(logits, targets)

            losses.update(loss.item(), args.batch_size)

            metrics = accuracy(logits, targets, topk=(1, 5))

            top1.update(metrics[0].item(), args.batch_size)
            top5.update(metrics[1].item(), args.batch_size)

    loss_dict = {
        "loss_ce": losses.avg
    }
    acc_dict = {
        'top1': top1.avg,
        'top5': top5.avg,
    }
    return loss_dict, acc_dict




