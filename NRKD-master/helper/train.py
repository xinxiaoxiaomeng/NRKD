from .utils import *


def train(args, train_loader, module_list, optimizer, criterion_list):

    for module in module_list:
        module.train()

    module_list[-1].eval()
    student = module_list[0]
    teacher = module_list[-1]

    criterion_ce = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_kd = criterion_list[2]

    losses = AverageMeter()
    loss_1 = AverageMeter()
    loss_2 = AverageMeter()
    loss_3 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print("训练：")
    for batch_idx, data in enumerate(train_loader):

        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()
        feat_s, logits_s = student(inputs, is_feat=True)

        with torch.no_grad():
            feat_t, logits_t = teacher(inputs, is_feat=True)
            feat_t = [f_t.detach() for f_t in feat_t]
        loss_ce = criterion_ce(logits_s, targets)
        loss_kl = criterion_kl(logits_s, logits_t)
        loss_kd = torch.FloatTensor([0.]).cuda()
        if args.distiller == 'nkd':
            feat_s = feat_s[1:-1]
            feat_t = feat_t[1:-1]
            loss_kd = criterion_kd(feat_s, feat_t, logits_s, logits_t)

        loss_ce = args.alpha * loss_ce
        loss_kl = args.beta * loss_kl
        loss_kd = args.gamma * loss_kd
        loss = loss_ce + loss_kl + loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), args.batch_size)
        loss_1.update(loss_ce.item(), args.batch_size)
        if loss_kl != 0:
            loss_2.update(loss_kl.item(), args.batch_size)
        if loss_kd != 0:
            loss_3.update(loss_kd.item(), args.batch_size)
            # print(loss_3.avg)
        metrics = accuracy(logits_s, targets, topk=(1, 5))
        top1.update(metrics[0].item(), args.batch_size)
        top5.update(metrics[1].item(), args.batch_size)

    loss_dict = {
        'losses': losses.avg,
        'loss_ce': loss_1.avg,
        'loss_kl': loss_2.avg,
        'loss_kd': loss_3.avg
    }

    acc_dict = {
        'top1': top1.avg,
        'top5': top5.avg
    }

    return loss_dict, acc_dict






























