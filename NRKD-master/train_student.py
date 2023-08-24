# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os
import argparse
import shutil
from models.get_model import get_model
from distillers import *
from helper import *
import torch.backends.cudnn as cudnn
from dataset import get_dataloader
import tensorboard_logger
import os



def parse_option():
    # 创建解析器，创建一个argparse对象
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--teacher', type=str, default='resnet20')
    parser.add_argument('--student', type=str, default='resnet8',)
    # cls
    parser.add_argument('--checkpoints', type=str, default='')
    # metric
    parser.add_argument('--pre_t', type=str, default='')
    parser.add_argument('--pre_s', type=str, default='')
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--temperature', default=4, type=float)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default="150, 180, 210", help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for loss_ce')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for loss_kd')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight for loss_nd')
    parser.add_argument('--distiller', type=str, default='nkd')
    # gpu
    parser.add_argument('--gpu', type=str, default='0')
    # nd
    parser.add_argument('--k', type=int, default=1, help='num of neighbours')
    # 解析命令行参数
    args = parser.parse_args()

    # 打印获取的参数
    print(args)

    # 将学习衰减的epoch用int型数据表示
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.student.lower() in ['mobilenetv2', 'shufflenetv1', 'shufflenetv2']:
        args.learning_rate = 0.01

    num = torch.rand((1, 1))
    # 当前训练的名称
    args.experiment = '{teacher}_{student}_{lr}_{dataset}_{distiller}_{sec}'.format(teacher=args.teacher,
                                                                                 student=args.student,
                                                                                 lr=args.learning_rate,
                                                                                 dataset=args.dataset,
                                                                                 distiller=args.distiller,
                                                                                 sec=str(num.item()))
    # 保存模型的路径
    args.trained_path = './trained'

    # 保存日志的路径
    args.log_path = 'log'

    args.save_folder = os.path.join(args.trained_path, args.experiment)

    return args


def main():

    args = parse_option()

    if args.gpu != '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 创建实验文件
    os.makedirs(args.save_folder)

    # 复制
    distiller = args.distiller.upper()
    if args.distiller == 'crd':
        shutil.copyfile('distillers/crd/criterion.py', args.save_folder + '/' + distiller + '.py')
    else:
        shutil.copyfile('distillers/'+distiller+'.py', args.save_folder+'/'+distiller+'.py')

    train_cls(args)


def train_cls(args):
    student_name = args.student.lower()
    teacher_name = args.teacher.lower()

    train_loader, test_loader, n_data, num_classes = get_dataloader(args)
    args.n_data = n_data

    student = get_model(student_name, args.dataset, num_classes)
    teacher = get_model(teacher_name, args.dataset, num_classes)

    # 加载教师模型
    teacher.load_state_dict(torch.load(args.checkpoints)["model"])
    if args.dataset == "cifar100" or args.distiller == "cifar10":
        data = torch.randn(2, 3, 32, 32)
    elif args.dataset == "tiny_imagenet":
        data = torch.randn(2, 3, 64, 64)
    elif args.dataset != "imagenet":
        data = torch.randn(2, 3, 224, 224)
    teacher.eval()
    student.eval()
    feat_t, _ = teacher(data, is_feat=True)
    feat_s, _ = student(data, is_feat=True)

    trainable_list = nn.ModuleList([])
    trainable_list.append(student)
    module_list = nn.ModuleList([])
    module_list.append(student)

    student.cuda()
    teacher.cuda()

    if args.distiller == 'kd':
        criterion_kd = None
    elif args.distiller == 'nkd':
        criterion_kd = NKDLoss(k=args.k)
    else:
        raise NotImplemented(args.distiller)

    # 损失函数
    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = KD(T=args.temperature)
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_kl)
    criterion_list.append(criterion_kd)

    module_list.append(teacher)

    # 使用gpu
    criterion_list.cuda()
    module_list.cuda()
    cudnn.benchmark = True

    optimizer = optim.SGD(trainable_list.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    with open(args.save_folder + '/log.txt', 'a') as f:
        f.write(str(args) + '\n')

    logger = tensorboard_logger.Logger(logdir=args.save_folder, flush_secs=2)
    # 训练

    mean_time = AverageMeter()

    best_acc = 0

    for epoch in range(1, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        start = time.time()
        print("[Epoch: %d]" % epoch)
        train_loss, train_acc = train(args, train_loader, module_list, optimizer, criterion_list)
        print('train_loss:', '%.2f' % train_loss["losses"], '\t train_acc:', '%.2f' % train_acc["top1"])

        test_loss, test_acc = test_cls(student, test_loader, criterion_ce, args)
        print("test_loss:", '%.2f' % test_loss["loss_ce"] + "\t test_acc:", '%.2f' % test_acc["top1"])
        # print(len(rec))
        cost_time = time.time() - start
        mean_time.update(cost_time)

        logger.log_value('train_acc', train_acc["top1"], epoch)
        logger.log_value('train_loss', train_loss["losses"], epoch)

        logger.log_value('test_acc', test_acc["top1"], epoch)
        logger.log_value('test_loss', test_loss["loss_ce"], epoch)
        logger.log_value('test_acc5', test_acc["top5"], epoch)

        # 记录
        with open(args.save_folder + '/log.txt', 'a') as f:
            f.write('\n\n epoch:' + str(epoch) + '\tlr:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
            f.write('\n train_loss:' + str('%.2f' % train_loss["losses"]) + '\t loss_ce:' + str(
                '%.2f' % train_loss["loss_ce"]) +
                    '\t loss_kl:' + str('%.2f' % train_loss["loss_kl"]) + "\t loss_kd:" + str(
                '%.2f' % train_loss["loss_kd"]))
            f.write('\n train_acc1:' + str('%.2f' % train_acc["top1"]) + '\t train_acc5:' + str(
                '%.2f' % train_acc["top5"]) + '\t time:' + str('%.2f' % (cost_time)))
            f.write('\n test_loss:' + str('%.2f' % test_loss["loss_ce"]) + '\t test_acc1:' + str(
                '%.2f' % test_acc["top1"]) + '\t test_acc5:' + str('%.2f' % test_acc["top5"]))


        if test_acc["top1"] > best_acc:
            best_acc = test_acc["top1"]
            state = {
                'epoch': epoch,
                'model': student.state_dict(),
                'accuracy': best_acc,
            }

            save_file = os.path.join(args.save_folder,
                                     '{teacher}_{student}_{dataset}_best.pth'.format(teacher=args.teacher,
                                                                                     student=args.student,
                                                                                     dataset=args.dataset))
            torch.save(state, save_file)

    print('best_acc:', best_acc)
    with open(args.save_folder + '/log.txt', 'a') as f:
        # 记录最好的精度
        f.write('\r\n' + 'best_acc:' + str(best_acc) + '\t time:' + str(mean_time.avg))

    args.experiment = '{teacher}_{student}_{lr}_{dataset}_{distiller}_{acc}'.format(teacher=args.teacher,
                                                                                    student=args.student,
                                                                                    lr=args.learning_rate,
                                                                                    dataset=args.dataset,
                                                                                    distiller=args.distiller,
                                                                                    acc='%.2f' % best_acc)
    os.rename(args.save_folder, os.path.join(args.trained_path, args.experiment))


if __name__ == '__main__':
    main()














