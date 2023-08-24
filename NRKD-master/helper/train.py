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


























def train_metric_kd(module_list, train_loader, criterion_list, optimizer, args):
    for module in module_list:
        module.train()

    module_list[-1].eval()
    student = module_list[0]
    teacher = module_list[-1]

    criterion_tr = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_kd = criterion_list[2]

    losses = AverageMeter()
    loss_1 = AverageMeter()
    loss_2 = AverageMeter()
    loss_3 = AverageMeter()

    print("训练：")
    for batch_idx, data in enumerate(train_loader):
        if args.distiller == 'crd':
            inputs, targets, index, contrast_idx = data
            inputs, targets, index, contrast_idx = inputs.cuda(), targets.cuda(), index.cuda(), contrast_idx.cuda()
        else:
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()

        feat_s, logits_s = student(inputs, is_feat=True)

        #
        with torch.no_grad():
            feat_t, logits_t = teacher(inputs, is_feat=True)
            feat_t = [f_t.detach() for f_t in feat_t]
        #
        loss_tr = criterion_tr(logits_s, targets)
        loss_kl = 0.0
        weight_div = 10.0
        loss_kd = 0.0
        weight_tr = 10.0
        if args.distiller == 'kd':
            loss_kl = criterion_kl(logits_s, logits_t)
            loss_kd = 0
            # weight_div = 0.9
            # weight_ce = 0.1

        elif args.distiller == 'nkd':
            feat_s = feat_s[1:-1]
            feat_t = feat_t[1:-1]
            # regress = module_list[1]
            # f_s = [reg(f) for (f, reg) in zip(f_s[:-1], regress)]

            loss_kd = criterion_kd(feat_s, feat_t, logits_s, logits_t)
            loss_kl = criterion_kl(logits_s, logits_t)

        elif args.distiller == 'dist':
            loss_kd = criterion_kd(logits_s, logits_t)

        elif args.distiller == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)

        elif args.distiller == 'sp':  # w / o kl  weight = 3000
            g_s = [feat_s[-1]]
            g_t = [feat_t[-1]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)

        elif args.distiller == 'fitnet':
            f_s = feat_s[1]
            f_t = feat_t[1]
            regress = module_list[1]
            f_s = regress(f_s)
            loss_kd = criterion_kd(f_s, f_t)

        elif args.distiller == 'at':  # include kl
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)

        elif args.distiller == 'ickd':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kl = criterion_kl(logits_s, logits_t)
            loss_kd = sum(loss_group)

        elif args.distiller == 'afd':
            feat_s = get_afd_fea(feat_s, args.student)
            feat_t = get_afd_fea(feat_t, args.teacher)
            loss_kd = criterion_kd(feat_s, feat_t)

        elif args.distiller == 'mgd':
            g_s = feat_s[-2]
            g_t = feat_t[-2]
            loss_kd = criterion_kd(g_s, g_t)

        elif args.distiller == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

        elif args.distiller == 'fkd':
            g_s = feat_s[-1]
            g_t = feat_t[-1]
            loss_kd = criterion_kd(g_s, g_t)

        elif args.distiller == 'cc':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            # weight_tt = 0.5
            weight_div = 0.5
            loss_kl = criterion_kl(logits_s, logits_t)
            loss_kd = criterion_kd(f_s, f_t)

        elif args.distiller == 'irg':
            transform_s = [feat_s[i] for i in args.transform_layer_s]
            transform_t = [feat_t[i] for i in args.transform_layer_t]
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, transform_s, transform_t, args.no_edge_transform)

        loss = weight_tr * loss_tr + weight_div * loss_kl + kd_weight[args.distiller] * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), args.batch_size)
        loss_1.update(loss_tr.item(), args.batch_size)
        if loss_kl != 0:
            loss_2.update(loss_kl.item(), args.batch_size)
        if loss_kd != 0:
            loss_3.update(loss_kd.item(), args.batch_size)
            # print(loss_3.avg)
        # break

    loss_dict = {
        'losses': losses.avg,
        'loss_tr': weight_tr * loss_1.avg,
        'loss_kl': weight_div * loss_2.avg,
        'loss_kd': kd_weight[args.distiller] * loss_3.avg
    }

    return loss_dict

def train_metric_triplet(model, train_loader, criterion, optimizer):
    model.train()
    loss_all, norm_all = [], []
    # train_iter = tqdm(train_loader, ncols=80)
    print("训练：")
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        fea, embeddings = model(images, is_feat=True)
        loss = criterion(embeddings, labels)
        loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #     train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    # print('[Epoch %d] Loss: %.5f\n' % (ep, torch.Tensor(loss_all).mean()))
    return torch.Tensor(loss_all).mean()
# 执行一个epoch的训练

def train_vanilla(model, train_loader, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 使用的损失函数
    criterion_ce = criterion

    model.train()
    print("训练：")
    for inputs, targets in train_loader:
        # print(inputs.size())
        # print(targets.size())
        # 使用GPU
        inputs, targets = inputs.cuda(), targets.cuda()
        # 获得输出
        outputs = model(inputs)
        # print(outputs.size())
        # 与GT的损失
        loss_ce = criterion_ce(outputs, targets)

        # 当前批次的损失值
        loss = loss_ce
        # 优化器的梯度置0
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        # 计算学生网络当前epoch的top1精度， top5精度
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        # 计算当前epoch的损失和精度
        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

    loss_dict = {
        "loss_ce": losses.avg
    }
    acc_dict = {
        "top1": top1.avg,
        "top5": top5.avg,
    }
    return loss_dict, acc_dict










