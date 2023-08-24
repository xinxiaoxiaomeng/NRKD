import torch
import torch.nn as nn
import argparse

import torch.optim as optim
import torch.backends.cudnn as cudnn

import time
import os
from models.get_model import get_model
from helper import *
from dataset import get_dataloader


def parse_option():
    # 创建解析器，创建一个argparse对象
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--model', type=str, default='resnet20')
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--checkpoints', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default="150, 180, 210", help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    # gpu
    parser.add_argument('--gpu', type=str, default='0')
    # 解析命令行参数
    args = parser.parse_args()

    # 将学习衰减的epoch用int型数据表示
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.dataset = args.dataset.lower()

    # 学习率调整
    if args.model in ['mobilenetv2', 'shufflenetv1', 'shufflenetv2']:
        args.learning_rate = 0.01

    args.experiment = '{model}_{lr}_{dataset}_{epoch}_{batch}'.format(model=args.model,
                                                                      lr=args.learning_rate,
                                                                      dataset=args.dataset,
                                                                      epoch=args.epochs,
                                                                      batch=args.batch_size)

    args.trained_path = './trained'

    args.save_folder = os.path.join(args.trained_path, args.experiment)

    args.distiller = ''
    print(args)

    return args


def main():
    args = parse_option()
    # 创建路径
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.gpu != '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    train_cls(args)


def train_cls(args):
    train_loader, test_loader, _, num_classes = get_dataloader(args)

    model = get_model(args.model, args.dataset, num_classes)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    model = model.cuda()
    criterion = criterion.cuda()
    # 大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True

    # 最好的精度
    best_acc = 0

    with open(args.save_folder + '/log.txt', 'a') as f:
        f.write(str(args) + '\n')

    # 训练
    for epoch in range(1, args.epochs + 1):
        # 根据当前epoch调整学习率
        adjust_learning_rate(epoch, args, optimizer)

        # 记录时间
        start = time.time()
        print("[Epoch: %d]" % epoch)
        # 训练模型
        train_loss, train_acc = train_vanilla(model, train_loader, criterion, optimizer)
        print('train_loss:', '%.2f' % train_loss["loss_ce"], '\t train_acc:', '%.2f' % train_acc["top1"])
        test_loss, test_acc = test_cls(model, test_loader, criterion, args)
        print("test_loss:", '%.2f' % test_loss["loss_ce"], '\t test_acc:', '%.2f' % test_acc["top1"])
        # print(rec)

        # 记录
        with open(args.save_folder + '/log.txt', 'a') as f:
            f.write('\n\n epoch:' + str(epoch) + '\t lr:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
            f.write('\n train_loss:' + str('%.2f' % train_loss["loss_ce"]) + '\t train_acc1:' + str(
                '%.2f' % train_acc["top1"]) + '\t train_acc5:' + str(
                '%.2f' % train_acc["top5"]) + '\t time:' + str('%.2f' % (time.time() - start)))
            f.write('\n test_loss:' + str('%.2f' % test_loss["loss_ce"]) + '\t test_acc1:' + str(
                '%.2f' % test_acc["top1"]) + '\t test_acc5:' + str('%.2f' % test_acc["top5"]))

        if test_acc["top1"] > best_acc:
            best_acc = test_acc["top1"]
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': best_acc,
            }

            save_file = os.path.join(args.save_folder,
                                     '{model}_{dataset}_best.pth'.format(model=args.model, dataset=args.dataset))
            torch.save(state, save_file)

    print('best_acc:', best_acc)

    with open(args.save_folder + '/log.txt', 'a') as f:
        # 记录最好的精度
        f.write('\r\n' + 'best_acc:' + str(best_acc))

if __name__ == '__main__':
    main()
